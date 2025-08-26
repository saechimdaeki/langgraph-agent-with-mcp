# langgraph-agent-with-mcp

## AI Agent란?
1. AI를 통해 의사결정 및 판단하는 프로그램
2. LLM Agent도 Agent
   -  의사결정의 도구로 LLM을 활용하는 Agent
   -  좋은 Agent를 만들려면 LLM을 잘 활용할 수 있어야 함

### LangGraph란?

```markdown
How are LangGraph and LangGraph Platform different?

LangGraph is a stateful, orchestration framework that brings added control to agent workflows.

LangGraph Platform is a service for deploying and scaling LangGraph applications, with an 

opinionated API for building agent UXs, plus an integrated developer studio.

```

- LangChain 기반의 orchestration framework 


### 프롬프트 작성 키워드
1. Succinct
2. 최대한 군더더기 없이 작성
3. 한번에 하나만 시켜라
   - 간단한 task는 더 작은 모델이 처리할 수 있고
   - 비용을 아끼고 답변 속도를 개선할 수 있음
 