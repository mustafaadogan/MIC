from tqdm import tqdm
from utils.util import write_results, check_answer, get_random_index_list

import torch


class MIC:
    def __init__(self):
        self.model = None
        self.processor = None
        self.results= {}
        self.device = None
        self.sc_exp_cnt = 1
        self.generation_cfg = {}
        self.output_file = None
        self.scoring_type = None
        

    def load_model(self, args) -> None:
        """
        Load the MIC model and processor.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """

        from model.instructblip import InstructBlipConfig, InstructBlipForConditionalGeneration, InstructBlipProcessor
        
        print('Loading MIC!!!')
        self.device = args.device
        self.scoring_type = args.scoring_type
        self.output_file = args.output_file
        self.sc_exp_cnt = args.sc_exp_cnt
        
        config = InstructBlipConfig.from_pretrained(args.hf_path)
        
        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            args.hf_path,
            config=config
        ).to(self.device, dtype=torch.bfloat16)
        

        image_palceholder="?"
        sp = [image_palceholder]+[f"<image{i}>" for i in range(20)]
        self.processor = InstructBlipProcessor.from_pretrained(
            args.lang_encoder_path
        )
        sp = sp + self.processor.tokenizer.additional_special_tokens[len(sp):]
        self.processor.tokenizer.add_special_tokens(
            {
            'additional_special_tokens': sp,
            }
        )

        if self.model.qformer.embeddings.word_embeddings.weight.shape[0] != len(self.processor.qformer_tokenizer):
            self.model.qformer.resize_token_embeddings(len(self.processor.qformer_tokenizer))
        
        self.replace_token="".join(32*[image_palceholder])
        
        self.generation_cfg = {
            'do_sample': False,
            'min_length': 1,
            'set_min_padding_size': False
        }
        
        self.generation_cfg['max_length'] = 4096


        print('MIC loaded!!!')

    def calculate_generated_text(self, prompt, vision_x):
        """
        Calculate generated text given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        Tuple[str, str]: Tuple containing the raw and salt answer text.
        """
        
        if self.model is None or self.processor is None:
            raise AttributeError('Model or processor is not initialized. Call load_model first!')
        
        inputs = self.processor(images=vision_x, text=prompt, return_tensors="pt", max_length=self.generation_cfg['max_length'], truncation=True)

        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)
        inputs['img_mask'] = torch.tensor([[1 for i in range(len(vision_x))]])
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)

        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values = inputs['pixel_values'],
                input_ids = inputs['input_ids'],
                attention_mask = inputs['attention_mask'],
                img_mask = inputs['img_mask'],
                do_sample=self.generation_cfg['do_sample'],
                max_length=self.generation_cfg['max_length'],
                min_length=self.generation_cfg['min_length'],
                set_min_padding_size=self.generation_cfg['set_min_padding_size'],
            )
        
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        return generated_text


    def test(self, data):
        """
        Test the model on the given data using the specified scoring type.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.

        Returns:
        None
        """
        
        for item in tqdm(data):
            assert len(item['support_classes_image_list']) == len(item['support_classes_raw_texts']), "Image-Caption count mismatch!"
            sc_results = {}
            sc_results['raw_results'] = []
            sc_results['score_list'] = []
            initial_index_list = list(range(len(item['support_classes_image_list'])))
            index_list = get_random_index_list(initial_index_list, self.sc_exp_cnt)

            for indexes in index_list:
                support_class_image_list = [item['support_classes_image_list'][i] for i in indexes]
                support_class_text_list = [item['support_classes_raw_texts'][i] for i in indexes]
                cot_info_list = [item['cot_info'][i] for i in indexes]
                vision_x = support_class_image_list
                vision_x += [item['query_image']]

                prompt = ''
                for i in range(len(vision_x)):
                    prompt += f'image {i} is <image{i}>{self.replace_token},'

                prompt += '.'
                for i, (raw_texts, cot_info) in enumerate(zip(support_class_text_list, cot_info_list)):
                    support_caption, support_foil = raw_texts[0], raw_texts[1]
                    cot_caption, cot_foil, is_caption_example = cot_info[0], cot_info[1], cot_info[2]

                    if is_caption_example:
                        prompt += f"Question: {item['support_prompt']} {i} {support_caption} {cot_caption}\n"
                    else:
                        prompt += f"Question: {item['support_prompt']} {i} {support_foil} {cot_foil}\n"


                query_caption, query_foil, is_caption_query = item['query_raw_texts'][0], item['query_raw_texts'][1], item['query_raw_texts'][2]

                if is_caption_query:
                    prompt += f"Question: {item['query_prompt']} {len(vision_x)-1} {query_caption} Answer:"
                else:
                    prompt += f"Question: {item['query_prompt']} {len(vision_x)-1} {query_foil} Answer:"

                item_result = {}
                score = [0, 1]

                if self.scoring_type == 'generated_text':
                    salt_answer = self.calculate_generated_text([prompt], vision_x)
                    
                    score = check_answer(salt_answer, is_caption_query) 
                    
                    item_result = {
                        'scores': score,
                        'caption_order': is_caption_query,
                        'prompt': prompt,
                        'salt_answer': salt_answer
                    }
                                
                else:
                    raise NotImplementedError(f'{self.scoring_type} not implemented yet!')

                
                sc_results['raw_results'].append(item_result)
                sc_results['score_list'].append(score)

                if sc_results['score_list'].count([1, 0]) >= (self.sc_exp_cnt/2) or sc_results['score_list'].count([0, 1]) >= (self.sc_exp_cnt/2):
                    break

            

            if sc_results['score_list'].count([1, 0]) >= (self.sc_exp_cnt/2):
                final_result = [1, 0] 
            else:
                final_result = [0, 1]
                
            sc_results['scores'] = final_result
            item_id = item['item_id']
            self.results[item_id] = sc_results
    
    def prepare_results(self):
        """
        Prepare and write the results to a JSON file.

        Parameters:
        - None

        Returns:
        None
        """
        write_results(self.output_file, self.results)