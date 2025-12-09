import os
from openai import OpenAI
import base64

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

class BaseVLM:
    def __init__(self, api_key, model, system_message="You are a helpful assistant.", base_url="https://api.openai.com/v1", erp=""):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.system_message = system_message
        self.erp = erp
        os.environ["OPENAI_API_KEY"] = self.api_key
        os.environ["OPENAI_API_BASE"] = self.base_url

    def get_client(self):
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_API_BASE"],
        )

    def generate_messages(self, user_content):
        return [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

    def call_model(self, user_content):
        client = self.get_client()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        msgs = self.generate_messages(user_content)
        extra_body = {"erp": self.erp} if self.erp else {}

        response = client.chat.completions.create(
            model=self.model,
            messages=msgs,
            stream=False,
            extra_body=extra_body,
            extra_headers=headers
        )
        return response.choices[0].message.content if response.choices else None
    def call_model_with_image(self, user_content, image_path):
        base64_image = encode_image(image_path)
        if base64_image:
            user_content = [
                {
                    "type": "text",
                    "text": user_content
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        return self.call_model(user_content)
class HistoryVLM(BaseVLM):
    def __init__(self, api_key, model, system_message, base_url, erp):
        super().__init__(api_key, model, system_message, base_url, erp)
        self.chat_history = []

    def call_model(self, user_content):
        self.chat_history.extend(self.generate_messages(user_content))
        client = self.get_client()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        extra_body = {"erp": self.erp} if self.erp else {}

        response = client.chat.completions.create(
            model=self.model,
            messages=self.chat_history,
            stream=False,
            extra_body=extra_body,
            extra_headers=headers
        )
        self.chat_history.append({
            "role": "assistant",
            "content": response.choices[0].message.content
        })
        return response

class MultiVLM(BaseVLM):
    def __init__(self, api_key, model, system_message, base_url, erp = ""):
        super().__init__(api_key, model, system_message, base_url, erp)

    
    def generate_messages_mul(self, content, image_urls):
        user_content = []
        for image_url in image_urls:
            base64_image = encode_image(image_url)
            if base64_image:
                image_url = f"data:image/jpeg;base64,{base64_image}"
            else:
                print(f"Error encoding image: {image_url}")
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        user_content.append({
            "type": "text",
            "text": content
        })
        return user_content
    def call_model_with_multi(self, content, image_urls):
        user_content = self.generate_messages_mul(content, image_urls)
        return self.call_model(user_content)
    


