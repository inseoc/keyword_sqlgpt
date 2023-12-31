<h1 align="center">
  Keyword SQL GPT
</h1>

<p align="left">
  
키워드를 기반으로 MS-SQL 쿼리를 생성해주는 OpenAI 기반 챗봇
- langchain==0.0.297, openai==0.28.0 버전 사용
- 모델 gpt3.5 적용
- 2023.09.12 시작

https://github.com/louis-li/SqlGPT.git
기반으로 작업
  
</p>





<h1 align="center">
  ChatGPT with SQL Server by Azure OpenAI  💡
</h1>

<p align="center">
  <strong>Database Assistant with Azure OpenAI</strong>
</p>

<p align="left">
  <strong>SqlGPT with Azure OpenAI</strong> is a sample Question and Answering bot using Azure OpenAI. It's designed to demonstrate how to use your own data in SQL Server for QnA.

  This demo shows how GPT demo can perform logic by demonstrating 
  - plan for action
  - update its plan with new collected information until goal is achieved
  
  ![SqlServer_ThoughtProcess.jpg](asset/SqlSever_ThoughtProcess2.jpg)
</p>


## Project structure
- Application - flask app (main application) with a simple HTML as frontend

## QuickStart

Note: Make sure you have docker installed

1. Open dowload this repository with `git clone https://github.com/louis-li/SqlGPT.git`
2. in application folder, mv .env.sample .env
3. Edit .env file and add your Azure OpenAI key and end point
4. Edit SQL Server/database/username/password info in .env file
5. Run `docker-compose build && docker-compose up`
6. Navigate to http://localhost:5010/

To stop just run Ctrl + C

Built with [🦜️🔗 LangChain](https://github.com/hwchase17/langchain)

