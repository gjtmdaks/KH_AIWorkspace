"""
    MCP
     - AI 모델과 외부도구를 연결할때 사용하는 표준 통신 규격
     - MCP이전에는 AI에게 외부도구를 바인딩하기 위해서는, 각 서비스마다 서로다른 전용코드를
       직접 개발해야 했다.
     - MCP는 이러한 불편함을 해결하기 위해 서버-클라이언트 구조를 도입하여 외부도구 바인딩
       과정을 표준화했다.
"""
import asyncio
from pathlib import Path

from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

DOWNLOADS_DIR = Path(__file__).parent.parent / "downloads"

# mcp통신함수
async def _call_mcp_tool(tool_name:str, arguments:dict) -> str:
    server_params = StdioServerParameters(
        command="npx",
        args=["-y","@modelcontextprotocol/server-filesystem", str(DOWNLOADS_DIR)]
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return "\n".join(
                block.text for block in result.content if hasattr(block, "text")
            )
        
def _run_mcp(tool_name:str, arguments:dict) -> str:
    try:
        return asyncio.run(_call_mcp_tool(tool_name, arguments))
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(lambda : asyncio.run(_call_mcp_tool(tool_name, arguments)))
            return future.result()
        
@tool
def write_file(path:str, content:str) -> str:
    """
        지정된 절대경로에 파일을 생성하거나 덮어씁니다.
    """
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    return _run_mcp("write_file", {"path":path, "content":content})

@tool
def read_file(path:str) -> str:
    """
        지정된 경로의 파일 내용을 읽어서 반환하는 함수
    """
    return _run_mcp("read_file", {"path":path})

filesystem_tools = [write_file, read_file]