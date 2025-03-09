import openai
from rich import print as rprint
import time
from typing import Union
from openai import OpenAI

# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
}


def convert_messages_to_prompt(messages):
    """
    Converts a list of messages(for chat) to a prompt (for completion) for OpenAI's API.

    :param messages:
    :return: prompt
    """
    prompt = ""
    for message in messages:
        prompt += f"{message['content']}\n"

    return prompt


def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        # errors: tuple = (openai.error.RateLimitError,),
):
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay
        while True:
            # try:
            return func(*args, **kwargs)

            # except errors as e:
            #     print(e)
            #     num_retries += 1
            #     if num_retries > max_retries:
            #         raise Exception(
            #             f"Maximum number of retries ({max_retries}) exceeded."
            #         )
            #     delay *= exponential_base * (1 + jitter * random.random())
            #     time.sleep(delay)
            # except Exception as e:
            #     raise e

    return wrapper


class Module(object):
    """
    This module is responsible for communicating with GPTs.
    """

    def __init__(self,
                 role_messages,
                 model="gpt-3.5-turbo-0301",
                 retrival_method="recent_k",
                 K=3):
        '''
        args:
        use_similarity:
        dia_num: the num of dia use need retrival from dialog history
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K

        self.chat_model = True if "gpt" in self.model else False
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None
        if self.model == 'claude':
            self.client = anthropic.Client(
                api_key="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")

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)

    def get_cache(self) -> list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else:
                return []
        else:
            return None

    @property
    def query_messages(self) -> list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]

    @retry_with_exponential_backoff
    def query(self, key, stop=None, temperature=0.0, debug_mode='Y', trace=True):
        openai.api_key = key
        rec = self.K
        if trace == True:
            self.K = 0
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False:
            messages[len(messages) - 1][
                'content'] += " Based on the failure explanation and scene description, analyze and plan again."
        self.K = rec
        response = ""
        # print('\n\nmessages = \n\n{}\n\n'.format(messages))
        get_response = False
        retry_count = 0

        while not get_response:
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query GPT failed for over 3 times!")
                return {}
            if 'claude' in self.model:
                response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.0,
                    system=messages[0]['content'],  # <-- system prompt
                    messages=[messages[1]]
                )
            try:
                if self.model in ['text-davinci-003']:
                    prompt = convert_messages_to_prompt(messages)
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature=temperature,
                        max_tokens=256
                    )
                    time.sleep(10)
                elif 'gpt' in self.model:
                    while True:
                        try:
                            response = openai.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                stop=stop,
                                temperature=temperature,
                                max_tokens=256
                            )
                            test = response.choices[0].message.content
                            break
                        except:
                            print("try again!!!")
                            time.sleep(5)
                            continue
                elif 'deepseek' in self.model:
                    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
                    while True:
                        try:
                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=messages,
                                stream=False
                            )
                            test = response.choices[0].message.content
                            break
                        except:
                            print("try again!!!")
                            time.sleep(10)
                            continue
                elif 'claude' in self.model:
                    response = self.client.messages.create(
                        model="claude-2.1",
                        system=messages[0]['content'],  # <-- system prompt
                        messages=[messages[1]]
                    )
                else:
                    raise Exception(f"Model {self.model} not supported.")

                get_response = True

            except Exception as e:
                retry_count += 1
                rprint("[red][OPENAI ERROR][/red]:", e)
                time.sleep(20)
        return self.parse_response(response)

    def parse_response(self, response):
        if self.model == 'claude':
            return response.message
        elif self.model in ['text-davinci-003']:
            return response.choices[0].text
        elif self.model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314',
                            "deepseek-chat", "deepseek-reasoner"]:
            return response.choices[0].message.content

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE[self.model]
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')

    def reset(self):
        self.dialog_history_list = []