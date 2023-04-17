import clueai
from langchain.llms.base import LLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.prompt import PromptTemplate
import gradio as gr
import os


class ChatYuanLLM(LLM):
    API_KEY = os.environ["CHATYUAN_API_KEY"]
    cl = clueai.Client(API_KEY, check_api_key=True)
        
    @property
    def _llm_type(self):
        return "custom"
    
    def _call(self, prompt, stop):
        prediction = self.cl.generate(model_name='ChatYuan-large',
                                      prompt = prompt)
        return prediction.generations[0].text


class Chat:
    def __init__(self):
        self.llm = ChatYuanLLM()
        
        self.conversation = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory
        )
        
    @property
    def prompt(self):
        return PromptTemplate(
            input_variables=["history", "input"], 
            template="{history}\n用户: {input}\n小元: "
        )
    
    @property
    def memory(self):
        return ConversationBufferWindowMemory(
            k=10, human_prefix="用户", ai_prefix="小元"
        )

    def user_input(self, user_message, history):
        return "", history + [[user_message, None]]

    def bot_predict(self, history):
        text = history[-1][0]
        history[-1][1] = self.conversation.predict(input=text)
        return history

chat = Chat()

with gr.Blocks() as app:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Enter text and press enter")

    msg.submit(fn=chat.user_input, inputs=[msg, chatbot], outputs=[msg, chatbot]).then(
        fn=chat.bot_predict, inputs=chatbot, outputs=chatbot)

app.launch()