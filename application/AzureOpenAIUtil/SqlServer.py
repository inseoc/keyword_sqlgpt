import urllib, os
from AzureOpenAIUtil.agent.SqlServerAgent import create_sql_agent, SQLDatabaseToolkit, SQLDatabase, HtmlCallbackHandler
# from langchain.callbacks.base import CallbackManager
from langchain.callbacks.manager import CallbackManager
from langchain.llms.openai import AzureOpenAI
from langchain.agents import AgentExecutor

class SqlServer:
    cb_handler = HtmlCallbackHandler()
    cb_manager = CallbackManager(handlers=[cb_handler])

    def __init__(self, llm, Server, Database, Username, Password, port=1433, odbc_ver=18, topK=10, question='') -> None:
        
        odbc_conn = 'Driver={ODBC Driver '+ str(odbc_ver) + ' for SQL Server};Server=tcp:' + \
            Server + f';Database={Database};Uid={Username};Pwd={Password};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;'
        params = urllib.parse.quote_plus(odbc_conn) # url 값이 깨지지 않게 보완 및 변환시켜주는 함수(url 주소 내 한글이 있으면 특수문자 및 영어로 치환하는 등)
        self.conn_str = 'mssql+pyodbc:///?odbc_connect={}'.format(params)


        db = SQLDatabase.from_uri(self.conn_str)
        self.toolkit = SQLDatabaseToolkit(db=db, callback_manager=self.cb_manager)
        # print(deploy_name)
        self.agent_executor = create_sql_agent(llm,
                toolkit=self.toolkit,
                verbose=True,
                topK=topK,
                callback_manager=self.cb_manager,
                question=question
            )
        
    def run(self, text: str):

        answer =  self.agent_executor.run(text)
        thought_process = self.cb_handler.get_and_reset_log()
        return answer, thought_process
        