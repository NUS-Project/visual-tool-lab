import transformers
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor,AutoModelForCausalLM, AutoTokenizer
from qwen_vl_utils import process_vision_info

class Qwen_test:
    def __init__(self, model_path: str, device: str):
        """
        Initializes the QwenTest class with the specified model.

        Args:
            model_path (str): The path to the pretrained model to use.
            device (str): The device to run the model on (e.g., "cuda:0").
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a medical expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."},
            {"role": "user", "content": prompt}
        ]

        # Prepare the input text
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # Generate the output from the model
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=5120
        )

        # Trim the generated ids
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Decode the output to text
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class QwenVL_Test:
    def __init__(self, model_path: str, device: str):
        """
        Initializes the QwenTest class with the specified model.

        Args:
            model_path (str): The path to the pretrained model.
            device (str): The device to run the model on (e.g., "cuda").
        """
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=self.device,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def chat(self, description: str = "",image_path: str = None ) -> str:
        """
        Analyzes the provided image and/or generates a description based on the input text.

        Args:
            image_path (str): The file path of the image to analyze.
            description (str): The description prompt for the image.

        Returns:
            str: The generated description response from the model.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a medical expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."
            },
            {
                "role": "user",
                "content": description  # Only include the text description here
            }
        ]

        # Add image input if provided
        if image_path:
            messages[1]["content"] = [
                {"type": "image", "image": image_path},
                {"type": "text", "text": description}
            ]

        # Prepare the input data
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages) if image_path else (None, None)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to the model's device
        inputs = inputs.to(self.model.device)

        # Generate output from the model
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=5120,
                do_sample=True,
                temperature=0.7
            )

        # Decode the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0]

class Llama_test:
    def __init__(self, model_path: str):
        """
        Initializes the pirate chatbot with the specified model ID.
        
        Args:
            model_id (str): The path to the pretrained model.
        """
        self.model_path = model_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_path,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        # self.messages = [
        #     {"role": "system", "content": "You are a medical expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."}
        # ]

    def chat(self, user_input: str) -> str:
        """
        Generates a response from the pirate chatbot based on user input.
        
        Args:
            user_input (str): The message from the user.
        
        Returns:
            str: The chatbot's response.
        """
        # Append the user's message to the messages list
        # self.messages=({"role": "user", "content": user_input})
        messages = [
                        {"role": "system", "content": "You are a medical expert. Provide only the letter corresponding to your answer choice (A/B/C/D/E/F)."},
                        {"role": "user", "content": user_input},
                    ]

        # Generate output using the pipeline
        outputs = self.pipeline(
            messages,
            max_new_tokens=2560,
        )
        
        # Extract and return the latest response
        return outputs[0]["generated_text"][-1]["content"]