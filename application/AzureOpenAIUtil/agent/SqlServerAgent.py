"""SQL agent."""
from __future__ import annotations
from typing import Any, List, Optional, Sequence
import ast
import re

from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.mrkl.prompt import FORMAT_INSTRUCTIONS
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
# from langchain.sql_database import SQLDatabase
from langchain.input import print_text
from langchain.schema import AgentAction, AgentFinish, LLMResult
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Extra, Field, validator
import os
from langchain.chains.llm import LLMChain
from langchain.llms.openai import AzureOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools.base import BaseTool
from typing import Any, List, Optional

from langchain.agents.agent_toolkits.base import BaseToolkit
"""SQLAlchemy wrapper around a database."""


import warnings
from typing import Any, Iterable, List, Optional

from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable

SQL_PREFIX = """You are an agent designed to interact with a Microsoft Azure SQL database.
        Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results using SELECT TOP in SQL Server syntax.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        
        If you encounter "Invalid column name" error in Observation, Use 'JOIN' statement and Find proper column name.
        If you constantly encounter "Invalid column name" error in Observation, Use 'list_tables_sql_db' tool again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        If the question does not seem related to the database, just return "I don't know" as the answer.
        """

SQL_SUFFIX = """Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.
{agent_scratchpad}"""


QUERY_CHECKER = """
{query}
Double check the {dialect} query above for common mistakes, including:
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Using the JOIN syntax for finding proper columns

If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query."""


TABLE_SELECTOR = '''
[QUESTION]
{question}

When a user asks the 'QUESTION' above, look at the DB table name and the table description below to decide which table to use.
You can use more than one table.

The OUTPUT is the table name to be used and the data format must be 'str'.

Look at an example of the form of input and output and reflect them to return the OUTPUT for the INPUT.

Examples of input:
[Table name 1: Table description 1]
[Table name 2: Table description 2]
[Table name 3: Table description 3]

Example output:
"Table name 2, table name 3."

INPUT:
{table_desc}

OUTPUT:
'''

# QUERY_CHECKER = """
# {query}
# Double check the {dialect} query above for common mistakes, including:
# - Using NOT IN with NULL values
# - Using UNION when UNION ALL should have been used
# - Using BETWEEN for exclusive ranges
# - Data type mismatch in predicates
# - Properly quoting identifiers
# - Using the correct number of arguments for functions
# - Casting to the correct data type
# - Using the proper columns for joins

# If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query."""

def _format_index(index: dict) -> str:
    return (
        f'Name: {index["name"]}, Unique: {index["unique"]},'
        f' Columns: {str(index["column_names"])}'
    )


class SQLDatabase:
    """SQLAlchemy wrapper around a database."""

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        # sample_rows_in_table_info: int = 2,
        sample_rows_in_table_info: int = 0,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[dict] = None,
        view_support: Optional[bool] = False,
        excluded_schema: Optional[list[str]] = ['db_accessadmin', 'db_backupoperator', 'db_datareader', 'db_datawriter', 'db_ddladmin', 'db_denydatareader', 'db_denydatawriter', 'db_owner', 'db_securityadmin', 'db_ssisadmin', 'guest', 'INFORMATION_SCHEMA', 'sys', 'sysadmin', 'public']
    ):
        """Create engine from database URI."""
        self._engine = engine
        self._schema = schema
        if include_tables and ignore_tables:
            raise ValueError("Cannot specify both include_tables and ignore_tables")

        self._inspector = inspect(self._engine)
        

        if schema is None:
            self._all_tables = []
            schemas = set(self._inspector.get_schema_names()) - set(excluded_schema)
            for schema in schemas:
                print("schema: %s" % schema)
                self._all_tables += set([schema+'.'+ table_name for table_name in  self._inspector.get_table_names(schema=schema) ]
                    + ([schema+'.'+ view_name for view_name in self._inspector.get_view_names(schema=schema)] if view_support else [])
                )
                # for table_name in self._inspector.get_table_names(schema=schema):
                    
            
            self._all_tables = set(self._all_tables + self._inspector.get_view_names(schema=schema))
        else:
            # including view support by adding the views as well as tables to the all
            # tables list if view_support is True
            self._all_tables = set(
                self._inspector.get_table_names(schema=schema)
                + (self._inspector.get_view_names(schema=schema) if view_support else [])
            )

        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )

        # self._metadata = metadata or MetaData()
        # # including view support if view_support = true
        # self._metadata.reflect(
        #     views=view_support,
        #     bind=self._engine,
        #     # only=self._usable_tables,
        #     # schema=self._schema,
        # )

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> SQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        return cls(create_engine(database_uri, **_engine_args), **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables

    # def get_table_names(self) -> Iterable[str]:
    #     """Get names of tables available."""
    #     warnings.warn(
    #         "This method is deprecated - please use `get_usable_table_names`."
    #     )
    #     return self.get_usable_table_names()

    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [tbl for tbl  in set(all_table_names)]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # add create table command
            table_name = table.split('.')
            if len(table_name) == 2:
                schema = table_name[0]
                table_name = table_name[1]
            # metadata_obj = MetaData(schema=schema)
            # metadata_obj.reflect(bind=self._engine, only=[table_name])
            # create_table = str(CreateTable(metadata_obj.tables[table]).compile(self._engine))
            # table_info = f"{create_table.rstrip()}"
            # has_extra_info = (
            #     self._indexes_in_table_info or self._sample_rows_in_table_info
            # )
            # if has_extra_info:
            #     table_info += "\n\n/*"

            # 1. 한 테이블에 대한 모든 컬럼명, 데이터타입, max-length 를 조회하는 쿼리
            get_table_info_query = f"""
                SELECT c.name AS 'Column Name', t.name AS 'Data Type', c.max_length AS 'Max Length'
                FROM sys.columns c
                JOIN sys.types t ON c.user_type_id = t.user_type_id
                WHERE c.object_id =  object_id('{table}')
                """
            table_info = f"Table Name: {table}\n"
            columns = self.run_no_throw(get_table_info_query)   # 참고로 columns 의 type은 str 이다.
            table_info += "(Column Name,Data Type,Max Length)\n" 
            table_info += str(columns)
            # col_list = ast.literal_eval(columns)
            # for col in col_list:
            #     table_info += f"{col[0]}\t{col[1]}\t{col[2]}\n"
            # table_info += ")\n"

            # 2. 각 컬럼에 대한 설명을 조회하는 쿼리 // 23.09.26
            columns = re.sub(r'\[\]', '', columns)  # 양쪽 대괄호 제거
            columns = re.findall(r"\('(.*?)', '(.*?)', (\d+)\)", columns)   # [(Column Name,Data Type,Max Length)] 형식의 리스트로 변환
            table_info += f"\n(Column's description in {table}):\n"
            col_names = list(map(lambda x: x[0], columns))
            
            for col_name in col_names:
                get_col_desc_query = f"""
                SELECT cast(VALUE as nvarchar(100)) as column_description 
                FROM ::FN_LISTEXTENDEDPROPERTY(NULL, 'SCHEMA', 'dbo', 'TABLE', '{table[4:]}', 'COLUMN', '{col_name}')
                """
                
                col_desc = self.run_no_throw(get_col_desc_query)
                
                if col_desc != '[]':
                    col_desc = re.sub(r'[\'\(\)\[\]]', '', col_desc)[:-1]
                    table_info += f"{col_name} description: {col_desc}\n"

            # 3. 외래 키를 조회하는 쿼리
            get_table_fk_query = f"""
            SELECT fk.name AS NameOfForeignKey
                ,t.name AS FKTableName
                , pc.name AS FKColumn
                , rt.name AS ReferencedTable
                , c.name AS ReferencedColumn
                FROM sys.foreign_key_columns AS fkc
                INNER JOIN sys.foreign_keys AS fk ON fkc.constraint_object_id = fk.object_id
                INNER JOIN sys.tables AS t ON fkc.parent_object_id = t.object_id
                INNER JOIN sys.tables AS rt ON fkc.referenced_object_id = rt.object_id
                INNER JOIN sys.columns AS pc ON fkc.parent_object_id = pc.object_id
                AND fkc.parent_column_id = pc.column_id
                INNER JOIN sys.columns AS c ON fkc.referenced_object_id = c.object_id
                AND fkc.referenced_column_id = c.column_id
                where t.object_id=object_id('{table}')
            """
            fk_info = self.run_no_throw(get_table_fk_query)
            if fk_info != '[]':
                table_info += f"Foreign Key Info: {fk_info}\n"
            
            # 4. sample row 를 일정 개수만큼만 추출 / default : _sample_rows_in_table_info = 2
            if self._sample_rows_in_table_info:
                get_sample_rows_query = f"""
                SELECT TOP {self._sample_rows_in_table_info} *
                FROM {table}"""
                sample_rows = self.run_no_throw(get_sample_rows_query)
                table_info += f"\nSample Rows:\n{sample_rows}\n"
            tables.append(table_info)
        final_str = "\n\n".join(tables)
        return final_str
    

    def get_table_desc(self, table_names: Optional[List[str]] = None) -> str:

        all_table_names = self.get_usable_table_names()

        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [tbl for tbl  in set(all_table_names)]
        table_info = ''
        for table in meta_tables:

            # add create table command
            table_name = table.split('.')

            if len(table_name) == 2:
                table_name = table_name[1] 

            get_table_desc_query = f"""
                SELECT cast(A.value as nvarchar(200)) as description
                FROM  SYS.extended_properties A 
                    LEFT OUTER JOIN SYSOBJECTS B 
                    ON A.major_id = B.id
                WHERE B.name = '{table_name}'
                AND A.minor_id = '0'
                """

            tab_desc = self.run_no_throw(get_table_desc_query)

            if tab_desc != '[]':
                tab_desc = re.sub(r'[\'\(\)\[\]]', '', tab_desc)[:-1]
            
            else:
                raise print(f"{table}'s description data is None. Insert some data.")

            table_info += f"[{table} : {str(tab_desc)}]\n"

        breakpoint()

        return table_info


    def _get_table_indexes(self, table: Table) -> str:
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(map(_format_index, indexes))
        return f"Table Indexes:\n{indexes_formatted}"

    def _get_sample_rows(self, table: Table) -> str:
        # build the select command
        command = select([table]).limit(self._sample_rows_in_table_info)

        # save the columns in string format
        columns_str = "\t".join([col.name for col in table.columns])

        try:
            # get the sample rows
            with self._engine.connect() as connection:
                sample_rows = connection.execute(command)
                # shorten values in the sample rows
                sample_rows = list(
                    map(lambda ls: [str(i)[:100] for i in ls], sample_rows)
                )

            # save the sample rows in string format
            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])

        # in some dialects when there are no rows in the table a
        # 'ProgrammingError' is returned
        except ProgrammingError:
            sample_rows_str = ""

        return (
            f"{self._sample_rows_in_table_info} rows from {table.name} table:\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
        )

    def run(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                connection.exec_driver_sql(f"SET search_path TO {self._schema}")
            cursor = connection.execute(text(command))
            if cursor.returns_rows:
                if fetch == "all":
                    result = cursor.fetchall()
                elif fetch == "one":
                    result = cursor.fetchone()[0]
                else:
                    raise ValueError("Fetch parameter must be either 'one' or 'all'")
                return str(result)
        return ""

    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    def run_no_throw(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(command, fetch)
        except SQLAlchemyError as e:
            """Format the error message"""
            return f"Error: {e}"


def ch(text: str) -> str:
    s = text if isinstance(text, str) else str(text)
    return s.replace("<", "&lt;").replace(">", "&gt;").replace("\r", "").replace("\n", "<br>")

class HtmlCallbackHandler (BaseCallbackHandler):
    html: str = ""

    def get_and_reset_log(self) -> str:
        result = self.html
        self.html = ""
        return result
    
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        self.html += f"LLM prompts:<br>" + "<br>".join(ch(prompts)) + "<br>";

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Do nothing."""
        self.html += f"LLM prompts end:<br>" + "<br>".join(response) + "<br>";
        # pass

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        self.html += f"<span style='color:red'>LLM error: {ch(error)}</span><br>"

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        # class_name = serialized["name"]
        # self.html += f"Entering chain: {ch(class_name)}<br>"
        pass

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        # self.html += f"Finished chain<br>"
        pass

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        self.html += f"<span style='color:red'>Chain error: {ch(error)}</span><br>"

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        action: AgentAction,
        color: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Print out the log in specified color."""
        # self.html += f"<span style='font-weight:bold;;color:yellow'>{ch(action)}</span><br>"
        pass

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        self.html += f'<span style="font-weight:bold;color:yellow">{ch(observation_prefix)}</span><br>{ch(output)}<br><span style="font-weight:bold;;color:yellow">{ch(llm_prefix)}</span><br>'

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self.html += f"<br><span style='color:red'>Tool error: {ch(error)}</span><br>"

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Optional[str],
    ) -> None:
        """Run when agent ends."""
        self.html += f"<span'>{ch(text)}</span><br>"

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        self.html += f"<span '>{ch(finish.log)}</span><br>"

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        # print(action.log)
        self.html += f"<span '>{ch(action.log)}</span><br>"
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Do nothing."""
        pass


# 밑 코드부터는 Custom tools 가 등장하는데, description 에서는 how/when/why 가 명확하게 들어가야 한다.
class BaseSQLDatabaseTool(BaseModel):
    """Base tool for interacting with a SQL database."""

    db: SQLDatabase = Field(exclude=True)

    # Override BaseTool.Config to appease mypy
    # See https://github.com/pydantic/pydantic/issues/4173
    class Config(BaseTool.Config):
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True
        extra = Extra.forbid


class QuerySQLDataBaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for querying a SQL database."""

    name = "query_sql_db"
    description = """
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned. 
    If an error is returned, rewrite the query, check the query, and try again.
    """

    def _run(self, query: str) -> str:
        """Execute the query, return the results or an error message."""
        return self.db.run_no_throw(query)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("QuerySqlDbTool does not support async")


class InfoSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    """Tool for getting metadata about a SQL database.(a comma-separated)"""

    name = "schema_sql_db"
    # description = """
    # Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables.
    # Be sure that the tables actually exist by calling list_tables_sql_db first!
    
    # Example Input: "table1, table2, table3"
    # """
    description = """
    Input to this tool is a comma-separated list of tables, output is the schema for those tables.
    
    If you need to use JOIN statement or don't know about word '코드' in user question, Refer to the output of the 'schema_sql_db'.

    Example Input: "table1, table2, table3"

    [Output's Structure start]

    (Column Name,Data Type,Max Length)
    [(column name, type name, length), ...]

    (Column's description in Table name)
    column name 1: code A:code A's mean, code B:code B's mean, ...
    개인구분고객구분코드: 2:유망, 1:소관, 8:기타(MIG), ...
    column name 3: code C:code C's mean, code D:code D's mean, ...
    보험료납입상태코드: E:AM지점장, G:PFPGA, 1:도입, ...

    [END]

    Be sure that the tables actually exist by calling list_tables_sql_db first!
    """

    def _run(self, table_names: str) -> str:
        """Get the schema for tables in a comma-separated list."""
        return self.db.get_table_info_no_throw(table_names.split(", "))

    async def _arun(self, table_name: str) -> str:
        raise NotImplementedError("SchemaSqlDbTool does not support async")


class ListSQLDatabaseTool(BaseSQLDatabaseTool, BaseTool):
    # """Tool for getting tables names."""

    # name = "list_tables_sql_db"
    # description = "Input is an empty string, output is a comma separated list of tables and list in the database."

    # def _run(self, tool_input: str = "") -> str:
    #     """Get the schema for a specific table."""
    #     return ", ".join(self.db.get_usable_table_names())

    # async def _arun(self, tool_input: str = "") -> str:
    #     raise NotImplementedError("ListTablesSqlDbTool does not support async")
    

    '''
    23.09.27
    추후 테이블 수가 늘어나면서 긴 입력값으로 인한 token length 에러가 일어날 것으로 예상
    이를 대비하기 위해 테이블 선별 기능 추가
    1) 모든 테이블 리스트를 입력 받는다
    2) 테이블 선별을 위한 프롬프트 준비 및 필요 데이터 추출
    3) llm(ChatOpenAI)을 활용하여 프롬프트 입력 후 추론 값 반환 => QueryCheckerTool 참고
    4) 이전 list_tables_sql_db 의 output 과 같이 사용자 질의를 해결하기 위해 선별된 테이블을 리스트로 반환

    테이블에 대한 간단한 요약(keyword) 방식이나 자세한 정보(full-text)를 같이 제공함으로써 GPT의 성능을 올려보는 시도도 필요
    혹은 사용자가 설정하는 옵션으로 만들어 보는 것도 고려해봄직하다.
    '''
    name = 'list_tables_sql_db'
    description = """
    Input is an empty string, output is a comma separated list of extracted some tables associated with user questions.

    Use this tool to select the proper table to use for user's question.
    """

    llm_chain: LLMChain = Field(
        default_factory=lambda: LLMChain(
            llm=ChatOpenAI(temperature=0, engine=os.getenv('DEPLOYMENT_NAME')),
            prompt=PromptTemplate(
                template=TABLE_SELECTOR, input_variables=["question", "table_desc"]
            ),
        )
    ) 

    def __init__(self, db, callback_manager, question:str = ''):
        super().__init__(db=db, callback_manager=callback_manager)
        self.question = question

    def _run(self, tool_input: str = "") -> str:
        """Get the selected table list for correct answers."""
        table_list = self.db.get_usable_table_names()   # [dbo.table_name, ...]
        table_desc = self.db.get_table_desc(table_list)
      
        selected_tables = self.llm_chain.predict(question=self.question, table_desc=table_desc)
        ## 아마 selected_tables 은 리스트가 아닌 string 일 것이다. 바로 return 해도 되지만 리스트의 대괄호([])를 지우기 위해
        ## 우선 print문을 통해 값을 확인 후 리스트로 변환해줘야 한다.
        breakpoint()
        # return ", ".join(selected_tables)
        return selected_tables

    async def _arun(self, tool_input: str = "") -> str:
        raise NotImplementedError("ListTablesSqlDbTool does not support async")


class QueryCheckerTool(BaseSQLDatabaseTool, BaseTool):
    """Use an LLM to check if a query is correct.
    Adapted from https://www.patterns.app/blog/2023/01/18/crunchbot-sql-analyst-gpt/"""
    # print(os.getenv('OPENAI_API_KEY'), type(os.getenv('OPENAI_API_KEY')))
    template: str = QUERY_CHECKER
    llm_chain: LLMChain = Field(
        default_factory=lambda: LLMChain(
            # llm=AzureOpenAI(temperature=0, deployment_name=os.getenv('DEPLOYMENT_NAME'))
            llm=ChatOpenAI(temperature=0, engine=os.getenv('DEPLOYMENT_NAME')),
            prompt=PromptTemplate(
                template=QUERY_CHECKER, input_variables=["query", "dialect"]
            ),
        )
    )
    name = "query_checker_sql_db"
    description = """
    Use this tool to double check if your query is correct before executing it.
    Always use this tool before executing a query with query_sql_db!
    """

    @validator("llm_chain")
    def validate_llm_chain_input_variables(cls, llm_chain: LLMChain) -> LLMChain:
        """Make sure the LLM chain has the correct input variables."""
        if llm_chain.prompt.input_variables != ["query", "dialect"]:
            raise ValueError(
                "LLM chain for QueryCheckerTool must have input variables ['query', 'dialect']"
            )
        return llm_chain

    def _run(self, query: str) -> str:
        """Use the LLM to check the query."""
        return self.llm_chain.predict(query=query, dialect=self.db.dialect)

    async def _arun(self, query: str) -> str:
        return await self.llm_chain.apredict(query=query, dialect=self.db.dialect)



class SQLDatabaseToolkit(BaseToolkit):
    """Toolkit for interacting with SQL databases."""

    db: SQLDatabase = Field(exclude=True)
    callback_manager: Optional[BaseCallbackManager] = None,
    
    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self.db.dialect

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    # def get_tools(self) -> List[BaseTool]:
    def get_tools(self, question) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            QuerySQLDataBaseTool(db=self.db, callback_manager=self.callback_manager),
            InfoSQLDatabaseTool(db=self.db, callback_manager=self.callback_manager),
            ListSQLDatabaseTool(db=self.db, callback_manager=self.callback_manager, question=question),
            # ListSQLDatabaseTool(db=self.db, callback_manager=self.callback_manager),
            QueryCheckerTool(db=self.db, callback_manager=self.callback_manager),
        ]


def create_prompt(
        tools: Sequence[BaseTool],
        prefix: str = SQL_PREFIX,
        suffix: str = SQL_SUFFIX,
        format_instructions: str = FORMAT_INSTRUCTIONS,
        input_variables: Optional[List[str]] = None,
    ):

    tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = FORMAT_INSTRUCTIONS.format(tool_names=tool_names)
    template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])

    if input_variables is None:
        input_variables = ["input", "agent_scratchpad"]
    
    return PromptTemplate(template=template, input_variables=input_variables)


def create_sql_agent(
    llm: BaseLLM,
    toolkit: SQLDatabaseToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: str = SQL_PREFIX,
    suffix: str = SQL_SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    early_stopping_method: str = "force",
    verbose: bool = False,
    **kwargs: Any,
) -> AgentExecutor:
    """Construct a sql agent from an LLM and tools."""
    # tools = toolkit.get_tools()
    tools = toolkit.get_tools(kwargs['question'])
    toolkit_names = list(map(lambda x: x.name, tools))

    prefix = prefix.format(dialect=toolkit.dialect, top_k=top_k)

    '''
    23.09.21 환경구축 및 코드 분석 후 개발
    코드를 좀 더 눈에 익히기 위해 ZeroShotAgent.create_prompt 말고 직접 작성
    '''

    prompt = create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        format_instructions=format_instructions,
        input_variables=input_variables,
    )

    llm_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        # verbose=verbose,
        callback_manager=callback_manager,
    )
    
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=toolkit_names,callback_manager=callback_manager, **kwargs)

    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        early_stopping_method=early_stopping_method,
        callback_manager=callback_manager
    )