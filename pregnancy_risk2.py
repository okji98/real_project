import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular

# 한글 설정
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)  # 마이너스 기호 깨짐 방지

# 페이지 설정
st.set_page_config(
    page_title="산모 건강 위험도 분석",
    page_icon=":pregnant_woman:",
    layout="wide"
)

# 제목 및 소개
st.title("산모 건강 위험도 분석 및 예측 모델:family:")
st.markdown("""
산모의 건강 데이터를 분석하고 위험도(저위험, 중위험, 고위험)를 예측 했습니다.
            
건강 지표를 입력하여 위험도를 확인하고, 데이터 시각화를 통해 각 요소가 위험도에 미치는 영향을 확인할 수 있습니다.
""")

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('Updated_Maternal_Health_Risk_Data.csv')
    # 위험도를 숫자로 변환
    risk_mapping = {'low risk': 0, 'mid risk': 1, 'high risk': 2}
    df['RiskLevel_num'] = df['RiskLevel'].map(risk_mapping)
    return df

df = load_data()

# 탭 생성
tab1, tab2, tab3, tab4 = st.tabs(["데이터 탐색", "데이터 시각화", "모델 예측 결과", "위험도 예측"])

# 탭 1: 데이터 탐색
with tab1:
    
    # 데이터 샘플 표시
    st.subheader("사용 된 데이터 샘플")
    st.dataframe(df.head())
    
    # 데이터 통계 정보
    st.subheader("데이터의 총 통계 정보")
    st.dataframe(df.describe())
    
    # 위험도별 데이터 수
    st.subheader("위험도별 총 데이터 수")
    risk_counts = df['RiskLevel'].value_counts().reset_index()
    risk_counts.columns = ['위험도', '데이터 수']
    
    fig = px.pie(risk_counts, names='위험도', values='데이터 수', color='위험도',
                color_discrete_map={'low risk': 'green', 'mid risk': 'orange', 'high risk': 'red'},
                title='위험도별 데이터 분포도')
    st.plotly_chart(fig, use_container_width=True)

# 탭 2: 데이터 시각화
with tab2:
    st.header("데이터 연관 관계 시각화")
    
    # 시각화 1: 연령 분포와 위험도
    st.subheader("1. 연령 분포와 위험도")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Age', title='연령 분포',
                          labels={'Age': '연령', 'count': '빈도'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("위험도별 연령 분포")

        # 그래프 생성
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.kdeplot(data=df, x='Age', hue='RiskLevel', fill=True, palette={'low risk': 'green', 'mid risk': 'orange', 'high risk': 'red'})

        # 그래프 제목 및 레이블 설정
        ax.set_title("위험도별 연령 분포", fontsize=14)
        ax.set_xlabel("연령")
        ax.set_ylabel("밀도")

        # 출력
        st.pyplot(fig)
    
    st.markdown("""
    **설명**: 연령 분포를 보면, 대부분의 산모가 20대 초반에 집중되어 있으며, 두 번째는 40-50대에서 나타납니다.
    고령 산모일수록 고위험 그룹에 속할 가능성이 높아지는 경향이 있습니다.
    """)
    
    # 시각화 2: 혈압과 위험도
    st.subheader("2. 혈압과 위험도")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df, x='RiskLevel', y='SystolicBP', color='RiskLevel',
                    title='위험도별 수축기 혈압 분포',
                    labels={'SystolicBP': '수축기 혈압', 'RiskLevel': '위험도'},
                    color_discrete_map={'low risk': 'green', 'mid risk': 'orange', 'high risk': 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, x='RiskLevel', y='DiastolicBP', color='RiskLevel',
                    title='위험도별 이완기 혈압 분포',
                    labels={'DiastolicBP': '이완기 혈압', 'RiskLevel': '위험도'},
                    color_discrete_map={'low risk': 'green', 'mid risk': 'orange', 'high risk': 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **설명**: 수축기 혈압과 이완기 혈압 모두 위험도와 밀접한 관련이 있습니다. 
    고위험 그룹은 평균적으로 더 높은 혈압을 보이며, 특히 수축기 혈압이 130mmHg 이상, 이완기 혈압이 80mmHg 이상인 경우 
    고위험 그룹에 속할 가능성이 높아집니다.
    """)
    
    # 시각화 3: 체온과 위험도
    # 체온(°F)을 섭씨(°C)로 변환
    df_temp = df.copy()
    df_temp['BodyTemp_C'] = (df_temp['BodyTemp'] - 32) * 5 / 9

    st.subheader("3. 체온과 위험도")

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.kdeplot(data=df_temp, x='BodyTemp_C', hue='RiskLevel', fill=True, 
                palette={'low risk': 'green', 'mid risk': 'orange', 'high risk': 'red'})

    # 그래프 제목 및 레이블 설정
    ax.set_title("위험도별 체온 분포 (섭씨 °C)", fontsize=14)
    ax.set_xlabel("체온 (°C)")
    ax.set_ylabel("밀도")

    # 출력
    st.pyplot(fig)
    
    st.markdown("""
    **설명**: 체온이 높을수록 위험도가 증가하는 경향이 있습니다. 
    고위험 그룹은 정상 체온보다 약간 높은 체온을 보이는 경우가 많습니다.
    """)
    
    # 시각화 4: 심박수와 위험도
    st.subheader("4. 심박수와 위험도")
    
    fig = px.box(df, x='RiskLevel', y='HeartRate', color='RiskLevel',
                title='위험도별 심박수 분포',
                labels={'HeartRate': '심박수(bpm)', 'RiskLevel': '위험도'},
                color_discrete_map={'low risk': 'green', 'mid risk': 'orange', 'high risk': 'red'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **설명**: 심박수가 높을수록 위험도가 증가하는 경향이 있습니다. 
    고위험 그룹은 평균적으로 더 높은 심박수를 보입니다.
    """)
    
    # 시각화 5: BMI와 위험도
    st.subheader("5. BMI와 위험도")
    
    fig = px.box(df, x='RiskLevel', y='BMI', color='RiskLevel',
                title='위험도별 BMI 분포',
                labels={'BMI': 'BMI', 'RiskLevel': '위험도'},
                color_discrete_map={'low risk': 'green', 'mid risk': 'orange', 'high risk': 'red'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **설명**: BMI가 높을수록 위험도가 증가하는 경향이 있습니다. 
    고위험 그룹은 더 넓은 BMI 분포를 보이며, 특히 높은 BMI 값에서 고위험 경향이 두드러집니다.
    """)
    
    # 시각화 6: 임신 주수와 위험도
    st.subheader("6. 임신 주수와 위험도")
    
    fig = px.box(df, x='RiskLevel', y='GestationalAge', color='RiskLevel',
                title='위험도별 임신 주수 분포',
                labels={'GestationalAge': '임신 주수', 'RiskLevel': '위험도'},
                color_discrete_map={'low risk': 'green', 'mid risk': 'orange', 'high risk': 'red'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **설명**: 임신 주수 자체는 위험도와 강한 상관관계를 보이지 않습니다. 
    모든 위험 그룹에서 비슷한 분포를 보이며, 다른 건강 지표가 위험도에 더 큰 영향을 미치는 것으로 보입니다.
    """)

# 탭 3: 예측 모델
with tab3:
    st.header("모델 예측 결과")
    st.write("""
산모의 건강 데이터를 기반으로 위험도를 예측합니다.
- 정확도: 약 70%
- 저위험 예측 정확도: 65%
- 중위험 예측 정확도: 66%
- 고위험 예측 정확도: 86%
""")
    # 모델 학습
    @st.cache_resource
    def train_model():
        # 특성과 타겟 분리
        X = df.drop(['RiskLevel', 'RiskLevel_num'], axis=1)
        y = df['RiskLevel_num']
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 특성 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 랜덤포레스트 모델 학습
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # 모델 평가
        y_pred = model.predict(X_test_scaled)
        
        return model, scaler, X_train, X_test, y_train, y_test, y_pred
    
    model, scaler, X_train, X_test, y_train, y_test, y_pred = train_model()
    
    # 모델 성능 평가
    st.subheader("모델 성능 평가")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**모델 성능 확인**")
        report = classification_report(y_test, y_pred, target_names=['저위험', '중위험', '고위험'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)
    
    with col2:
        st.write("**혼동 행렬 확인**")
        st.markdown("""혼동 행렬을 통해 모델이 정확하게 예측한 사례의 수를 확인 할 수 있습니다.""")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['저위험', '중위험', '고위험'],
                   yticklabels=['저위험', '중위험', '고위험'])
        plt.xlabel('예측')
        plt.ylabel('실제')
        st.pyplot(fig)

    # 특성 중요도
    st.write("**특성 중요도 확인**")
    feature_importance = pd.DataFrame({
        '특성': X_train.columns,
        '중요도': model.feature_importances_
    }).sort_values('중요도', ascending=False)
    
    fig = px.bar(feature_importance, x='특성', y='중요도', 
                title='특성 중요도',
                labels={'특성': '특성', '중요도': '중요도'})
    st.plotly_chart(fig, use_container_width=True)
    
    
    # 설명기 생성
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns,
        class_names=['저위험', '중위험', '고위험'],
        mode='classification'
    )

    # 특정 인스턴스 설명
    exp = explainer.explain_instance(
        X_test.iloc[0].values, 
        model.predict_proba, 
        num_features=10
    )
    exp.show_in_notebook()
    st.markdown("""
    **모델 설명**:
    특성 중요도를 통해, 각 항목들이 미치는 영향을 파악 할 수 있습니다.
    이 중 가장 영향이 큰 것으로 혈당이 가장 중요한 요소이고,
                
    수축기 혈압, 이완기 혈압, 나이, BMI 등이 위험도 예측에 영향을 끼치는 요소임을 확인 할 수 있습니다.
    """)

# 사이드바에서 데이터 입력 받기
st.sidebar.header("사용자 데이터 입력")

# 사이드바에서 데이터 입력받기
age = st.sidebar.number_input("나이", min_value=20, max_value=70, value=25)
systolic_bp = st.sidebar.number_input("수축기 혈압 (mmHg)", min_value=70, max_value=180, value=120)
diastolic_bp = st.sidebar.number_input("이완기 혈압 (mmHg)", min_value=40, max_value=120, value=80)
bs = st.sidebar.number_input("혈당 (mmol/L)", min_value=5.0, max_value=20.0, value=7.0, step=0.1)
body_temp = st.sidebar.number_input("체온 (°C)", min_value=35.0, max_value=41.0, value=36.5, step=0.1) # 섭씨로 입력하기
body_temp_c = (body_temp * 9/5) + 32  # 화씨 -> 섭씨 변환
heart_rate = st.sidebar.number_input("심박수 (bpm)", min_value=60, max_value=100, value=75)
height = st.sidebar.number_input("키 (cm)", min_value=140.0, max_value=190.0, value=155.0, step=0.1)
weight = st.sidebar.number_input("체중 (kg)", min_value=30.0, max_value=100.0, value=45.0, step=0.1)
bmi = weight / ((height/100) ** 2)
gestational_age = st.sidebar.number_input("임신 주수", min_value=10, max_value=40, value=15)

# 예측 버튼 결과 저장 (초기값)
prediction = None

# 사이드바에서 예측 버튼 추가
if st.sidebar.button("위험도 예측"):
    # 입력 데이터 생성
    input_data = pd.DataFrame({
        'Age': [age],
        'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp],
        'BS': [bs],
        'BodyTemp': [body_temp_c],
        'HeartRate': [heart_rate],
        'Height(cm)': [height],
        'Weight(kg)': [weight],
        'BMI': [bmi],
        'GestationalAge': [gestational_age]
    })

    # BMI 기반 위험도 평가
    bmi_risk = 0  # 기본값: 저위험
    if bmi < 16:  # 심각한 저체중
        bmi_risk = 2  # 고위험
    elif bmi < 18.5:  # 저체중
        bmi_risk = 1  # 중위험
    elif 18.5 <= bmi < 23.0:
        bmi_risk = 0  # 저위험
    elif 23.0 <= bmi < 25.0:
        bmi_risk = 1  # 중위험
    elif bmi >= 27:  # 비만
        bmi_risk = 2  # 고위험
    
    # 입력 데이터 스케일링
    input_scaled = scaler.transform(input_data)
    
    # 모델 예측 수행
    model_prediction = model.predict(input_scaled)[0]
    
    # BMI 기반 위험도와 모델 예측 위험도 중 더 높은 값 선택
    prediction = max(model_prediction, bmi_risk)
    
with tab4:
    st.header('위험도 예측')
    # 예측 결과 표시
    risk_labels = ['저위험', '중위험', '고위험']
    risk_colors = ['green', 'orange', 'red']
        
    st.markdown("### 예측 결과")
    if prediction is not None:
        st.markdown(f"<h1 style='text-align: center; color: {risk_colors[prediction]};'>{risk_labels[prediction]}</h1>", unsafe_allow_html=True)
        
        # 예측 확률 계산 (원래 모델의 확률)
        proba = model.predict_proba(input_scaled)[0]

        # BMI 기반 위험도를 반영한 확률 조정
        if prediction != model_prediction:
            st.info(f"BMI({bmi:.1f})가 정상 범위를 벗어나 위험도가 조정되었습니다.")
                
            proba_df = pd.DataFrame({
                    '위험도': risk_labels,
                    '확률': proba * 100
                })
                
            # 확률 시각화
            fig = px.bar(proba_df, x='위험도', y='확률', color='위험도',
                            color_discrete_map={'저위험': 'green', '중위험': 'orange', '고위험': 'red'})
            fig.update_layout(title='위험도 예측 확률')
            st.plotly_chart(fig, use_container_width=True)
                
            # 위험 요소 분석 및 권장 사항
            st.subheader("위험 요소 분석 및 권장 사항")
                
            if prediction == 0:  # 저위험
                st.success("현재 산모의 건강 상태는 양호합니다. 정기적인 검진을 계속 받으시기 바랍니다.")
                st.markdown("""
                **주요 권장사항**:
                - 정기적인 산전 검진 유지
                - 균형 잡힌 식단 섭취
                - 적절한 운동 유지
                - 충분한 휴식과 수면
                """)
            elif prediction == 1:  # 중위험
                st.warning("일부 건강 지표에 주의가 필요합니다. 더 자주 검진을 받고 의사의 조언을 따르세요.")
                st.markdown("""
                **주요 권장사항**:
                - 검진 주기 단축 (2주마다 검진 권장)
                - 혈압 및 혈당 정기적 모니터링
                - 스트레스 관리 및 충분한 휴식
                - 의사가 권장하는 식이요법 준수
                """)
            else:  # 고위험
                st.error("여러 건강 지표에서 위험 신호가 감지되었습니다. 즉시 의료진과 상담하시기 바랍니다.")
                st.markdown("""
                **주요 권장사항**:
                - 즉시 전문 의료진 상담
                - 필요시 입원 치료 고려
                - 24시간 건강 상태 모니터링
                - 모든 활동 전 의사와 상담
                - 응급상황 대비 계획 수립
                """)
    else:
        st.info("위험도를 예측하려면 사이드바에서 데이터를 입력 후, '위험도 예측' 버튼을 클릭 해주세요.")   
