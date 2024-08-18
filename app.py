import yaml
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from lifecycle_functions_v1 import find_first_ipchun_revised, get_ipchun_opposite, get_ipchun_same, get_saju, get_solar_term
from lifecycle_functions_v1 import calculate_lifecycle, create_enhanced_3d_lifecycle_chart
from lifecycle_functions_v1 import load_saju_data
from dotenv import load_dotenv
from datetime import datetime
import os

# API KEY 정보 로드
load_dotenv()

# 전역 변수로 사주 데이터 로드 (성능 최적화)
SAJU_DATA = load_saju_data('saju_cal.csv')

def format_lifecycle(lifecycle_data):
    formatted = ""
    for i, (term, year, age) in enumerate(lifecycle_data):
        if i % 4 == 0 and i != 0:
            formatted += "\n"
        formatted += f"{term}({year:.1f}년, {age}세) | "
    return formatted.strip()

def parse_date(input_str):
    if len(input_str) == 12 and input_str.isdigit():
        return input_str
    return None

def create_chain(model_choice):
    prompt = load_prompt("saju_prompt_general.yaml", encoding="utf-8")
    
    if model_choice == "OpenAI GPT":
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    else:  # Google Gemini
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain

def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]

st.title("AI LIFE CYCLE 길잡이 💬")

with st.sidebar:
    if st.button("대화 초기화"):
        reset_session()
        st.rerun()  # 여기를 수정했습니다

    st.session_state.model_choice = st.selectbox(
        "AI 모델을 선택해 주세요", ("OpenAI GPT", "Google Gemini"), index=0
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False

if not st.session_state.analyzed:
    gender = st.radio("성별을 선택하세요:", ("남성", "여성"))
    birth_input = st.text_input("생년월일시를 입력하세요 (YYYYMMDDHHMMM 형식):")

    if st.button("분석 시작"):
        if gender and birth_input:
            parsed_date = parse_date(birth_input)
            if parsed_date:
                try:
                    birth_year = int(parsed_date[:4])
                    current_year = datetime.now().year

                    saju_result = get_saju(parsed_date, SAJU_DATA)
                    st.write(f"당신의 사주팔자: {saju_result}")

                    ipchun_same = get_ipchun_same(saju_result)
                    ipchun_opposite = get_ipchun_opposite(saju_result)
                    month_ground = saju_result.split(',')[0].split()[1][1]
                    solar_term = get_solar_term(month_ground)

                    analysis_result = f"""
                    입춘점 정보:
                    1. 일간월지를 그대로 사용한 입춘점: {ipchun_same}년
                       해당 절기: {solar_term}
                    2. 월지의 정반대 지지를 사용한 입춘점: {ipchun_opposite}년
                       해당 절기: {get_solar_term(ipchun_opposite[1])}
                    """

                    first_ipchun_same = find_first_ipchun_revised(birth_year, ipchun_same, SAJU_DATA)
                    first_ipchun_opposite = find_first_ipchun_revised(birth_year, ipchun_opposite, SAJU_DATA)

                    if first_ipchun_same:
                        analysis_result += f"\n당신의 생애 첫 순방향 입춘점: 입춘({first_ipchun_same[0]}, {first_ipchun_same[1]}세)"
                    if first_ipchun_opposite:
                        analysis_result += f"\n당신의 생애 첫 역방향 입춘점: 입춘({first_ipchun_opposite[0]}, {first_ipchun_opposite[1]}세)"

                    lifecycle_forward = calculate_lifecycle(ipchun_same, birth_year)
                    lifecycle_backward = calculate_lifecycle(ipchun_opposite, birth_year)

                    if lifecycle_forward:
                        analysis_result += "\n\n순방향 입춘점으로 계산한 60년 생애주기:\n"
                        analysis_result += format_lifecycle(lifecycle_forward)
                    if lifecycle_backward:
                        analysis_result += "\n\n역방향 입춘점으로 계산한 60년 생애주기:\n"
                        analysis_result += format_lifecycle(lifecycle_backward)

                    st.write(analysis_result)

                    st.subheader("생애주기 3D 차트")
                    
                    st.subheader("순방향 생애주기 3D 차트")
                    fig_forward = create_enhanced_3d_lifecycle_chart(lifecycle_forward, birth_year)
                    st.plotly_chart(fig_forward, use_container_width=True)

                    st.subheader("역방향 생애주기 3D 차트")
                    fig_backward = create_enhanced_3d_lifecycle_chart(lifecycle_backward, birth_year)
                    st.plotly_chart(fig_backward, use_container_width=True)

                    chain = create_chain(st.session_state.model_choice)
                    
                    lifecycle_str = format_lifecycle(lifecycle_backward)

                    initial_response = chain.invoke({
                        "saju": saju_result,
                        "lifecycle": lifecycle_str,
                        "birth_year": str(birth_year),
                        "current_year": str(current_year),
                        "gender": gender,
                        "question": "전체적인 사주 해석과 24절기에 기반한 60년 인생의 생애주기를 자세하게 분석해주세요."
                    })

                    st.subheader("AI의 사주 해석 및 조언")
                    st.write(initial_response)

                    st.session_state.analyzed = True
                    st.session_state.saju_result = saju_result
                    st.session_state.lifecycle_str = lifecycle_str
                    st.session_state.birth_year = birth_year
                    st.session_state.gender = gender
                    st.session_state.messages.append({"role": "assistant", "content": initial_response})

                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")
            else:
                st.error("올바른 형식으로 생년월일시를 입력해주세요.")
        else:
            st.error("성별과 생년월일시를 모두 입력해주세요.")

else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if st.session_state.analyzed:
    if prompt := st.chat_input("질문을 입력하세요:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        full_prompt = f"""
        당신은 30년 경력의 사주팔자 통변 전문가입니다. 다음 정보를 바탕으로 사용자의 질문에 답변해주세요:

        사용자의 사주팔자: {st.session_state.saju_result}
        생년: {st.session_state.birth_year}
        성별: {st.session_state.gender}
        
        사용자 질문: {prompt}
        
        답변 시 다음 사항을 고려해주세요:
        1. 사주팔자의 음양오행 균형을 고려하여 해석해주세요.
        2. 사용자의 성별과 나이에 맞는 분석을 제공해주세요.
        3. 현대적 맥락에서 실용적인 분석을 해주세요.
        4. 답변은 친절하고 이해하기 쉬운 언어로 분석해주세요.
        5. 필요하다면 생애주기 정보를 참고하여 답변할 수 있습니다.
        사주와 운세는 절대적인 것이 아니라 참고사항임을 언급하고, 개인의 노력과 선택이 중요함을 강조해주세요.
        """
        
        if st.session_state.model_choice == "OpenAI GPT":
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        else:  # Google Gemini
            llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = llm.predict(full_prompt)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
