import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI  # Ensure you have the OpenAI library installed
import pandas as pd
from fpdf import FPDF
from io import BytesIO

# Function to estimate relative blood volume
def get_VolRel(sex: str, body_fat: float) -> float:
    sex = sex.lower()
    
    if sex == "men":
        body_fat_values = np.array([4, 14, 15, 21, 22, 28, 29])
        body_water_values = np.array([70, 63, 63, 57, 57, 52, 52])
    elif sex == "women":
        body_fat_values = np.array([4, 20, 21, 29, 30, 36, 37])
        body_water_values = np.array([70, 58, 58, 52, 52, 45, 45])
    else:
        raise ValueError("Sex must be 'men' or 'women'.")

    return (np.interp(body_fat, body_fat_values, body_water_values) * .735) / 100

def get_MAP(VO2max, weight):
    """
    Calculate MAP (Maximum Aerobic Power) from VO2max and weight.
    
    Parameters:
    VO2max (float): Maximum oxygen uptake in ml/min/kg
    weight (float): Body weight in kg
    
    Returns:
    float: Maximum Aerobic Power (MAP)
    """
    return ((VO2max / 1000) * weight - 0.435) / 0.01141
# Function to generate PDF


# Streamlit UI
st.title("Metabolic Profile Analyzer")

sex = st.selectbox("Sex", ["Men", "Women"])
vo2max = st.slider("VO2max", 30, 90, 58)
fat = st.slider("Body Fat %", 4, 40, 10)
vlamax = st.slider("VLaMax", 0.1, 1.5, 0.8)
weight = st.slider("Weight (kg)", 50, 100, 75)

# Calculations
VolRel = get_VolRel(sex, fat)
Ks4 = 11.7 #weight * vo2max / 400
Ks1 = 0.25 ** 2
Ks2 = 1.1 ** 3
Ks3 = 0.02049 / VolRel
VO2ss = np.arange(1, vo2max - 5, 0.01)
ADP = np.sqrt((Ks1 * VO2ss) / (vo2max - VO2ss))
vLass = 60 * vlamax / (1 + (Ks2 / ADP ** 3))
LaComb = Ks3 * VO2ss
vLanet = abs(vLass - LaComb)
Intensity = ((vLass * (VolRel * weight) * ((1/4.3) * 22.4) / weight) + VO2ss) / (Ks4 / weight)


# Graphs
col1, col2 = st.columns(2)

with col1:
    fig1, ax = plt.subplots()
    ax.plot(Intensity, ((vLass * (VolRel * weight) * ((1/4.3) * 22.4) / weight) + VO2ss), label="Oxygen Demand", color='red')
    ax.plot(Intensity, VO2ss, label="Oxygen Uptake", color='blue')
    ax.fill_between(Intensity, ((vLass * (VolRel * weight) * ((1/4.3) * 22.4) / weight) + VO2ss), VO2ss, color="grey")
    ax.set_xlabel('Watts [W]')
    ax.set_ylabel('Oxygen [ml/min/kg]')
    ax.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.plot(Intensity, vLanet, label='Lactate Accumulation')
    ax2.plot(Intensity, vLass, label='Gross Lactate Formation')
    ax2.plot(Intensity, LaComb, label='Potential Lactate Removal')
    arg_sAT = np.argmin(vLanet)
    sAT = Intensity[arg_sAT]
    ax2.scatter(Intensity[arg_sAT], vLass[arg_sAT], color='red')
    ax2.legend()
    st.pyplot(fig2)

# Carbohydrate & Fat Utilization
CHO_util = vLass * (weight * VolRel) * 60 / 1000 / 2 * 162.14
Fat_util = (vLanet[:arg_sAT] * VolRel) / 0.01576 * weight * 60 * 4.65 / 9.5 / 1000

fig3, ax4 = plt.subplots(figsize=(10, 5))
ax4.plot(Intensity[:len(CHO_util)], CHO_util, label='Carbohydrates', color='darkgoldenrod')
ax4.set_xlabel('Power [W]')
ax4.set_ylabel('Carbohydrate g/h')
ax4.legend()
ax3 = ax4.twinx()
ax3.plot(Intensity[:len(Fat_util)], Fat_util * 9.5, label='Fat', color='green')
ax3.set_ylabel('Fat kcal/h')
ax3.legend()
st.pyplot(fig3)

# Compute metabolic key points
CarbMax = Intensity[np.min(np.where(CHO_util >= 90))]
CarbMax_fat = Fat_util[np.argmin(abs(Intensity - CarbMax))] * 9.5
CarbMax_carbs = CHO_util[np.argmin(abs(Intensity - CarbMax))]
CarbMax_vo2 = VO2ss[np.argmin(abs(Intensity - CarbMax))]

res = pd.DataFrame()
res = pd.DataFrame([Intensity, vLanet, VO2ss,CHO_util, Fat_util], index = ['Power','Lactate','Vo2','Cho','Fat'])
res = res.T
mini = res[res.index <= arg_sAT]#.Lactate.argmax()
lt1_zone = mini[mini.index>mini.Lactate.argmax()]
lt1 = lt1_zone[(lt1_zone.Lactate < lt1_zone.Lactate.max()-.25)].iloc[0]


# Summary Table
table_data = {
    'Metric': ['Power', 'Kcal/h', 'Carbs g/h', '% VO2max'],
    'FatMax': [res[res.index == res.Fat.argmax()].Power.iloc[0] , 
    res[res.index == res.Fat.argmax()].Fat.iloc[0] * 9.5, 
    res[res.index == res.Fat.argmax()].Cho.iloc[0], 
    np.round(res[res.index == res.Fat.argmax()].Vo2.iloc[0] / vo2max * 100,1)],
    'CarbMax': [res[res.Cho>90].iloc[0].Power.round(0), 
    res[res.Cho>90].iloc[0].Fat.round(1), 
    res[res.Cho>90].iloc[0].Cho.round(1), 
    np.round(res[res.Cho>90].iloc[0].Vo2 / vo2max * 100,1)],
    'Lt1':[lt1.Power.round(0), lt1.Fat.round(1), lt1.Cho.round(1), np.round(lt1.Vo2/vo2max*100,1)],
    
    'MLSS': [np.round(sAT,0), 0, np.round(CHO_util[arg_sAT],1), np.round((res.loc[arg_sAT].Vo2/vo2max)*100,1)],
    'Vo2max': [get_MAP(vo2max, weight), 0, res[res.Power>=vo2max/(Ks4/weight)].iloc[0].Cho,100]
    
}

st.table(table_data)

# LLM Commentary using OpenAI GPT API
st.subheader("Performance Analysis")

#OpenAI.api_key = "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"

prompt = f"""
Analyze the following metabolic profile for an athlete:

- VO2max: {vo2max} ml/min/kg
- VLaMax: {vlamax} mmol/L/s
- FatMax occurs at {res[res.index == res.Fat.argmax()].Power.iloc[0]} W with this value of consommation {np.max(Fat_util) * 9.5}
- Lt1 is at {lt1.Power.round(0):} W
- Lt2 (Max Lactate Steady State) occurs at {sAT:.1f} W  


1) Compare the values to the litterature and normative value for elite athletes, especially VO2max, Lt2, fatmax Watts and fat oxydation
2) Explain strengths and weaknesses of this athlete for a long distance cycling and areas to train 
3) Give the profile of the cyclist : more punchy or endurance rider and which race (Grand tour, classic, sprints, time trial) suits him the most 

do it in maximum 250 words
"""

#client = OpenAI(api_key = "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")

client = OpenAI(api_key = "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")



def generate_pdf(figures, table_data, analysis_text):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Metabolic Profile Analysis", ln=True, align='C')
    pdf.ln(5)

    # Save the first two figures in the same row
    img_buffer1 = BytesIO()
    figures[0].savefig(img_buffer1, format='png')
    img_buffer1.seek(0)
    img_path1 = "plot_0.png"
    with open(img_path1, "wb") as f:
        f.write(img_buffer1.getvalue())
    
    img_buffer2 = BytesIO()
    figures[1].savefig(img_buffer2, format='png')
    img_buffer2.seek(0)
    img_path2 = "plot_1.png"
    with open(img_path2, "wb") as f:
        f.write(img_buffer2.getvalue())
    
    pdf.image(img_path1, x=10, y=30, w=80)
    pdf.image(img_path2, x=110, y=30, w=80)

    # Save and add the last figure (centered and below the first two)
    img_buffer3 = BytesIO()
    figures[2].savefig(img_buffer3, format='png')
    img_buffer3.seek(0)
    img_path3 = "plot_2.png"
    with open(img_path3, "wb") as f:
        f.write(img_buffer3.getvalue())
    
    pdf.image(img_path3, x=15, y=90, w=160)

    # Table Section
    pdf.ln(150)
    pdf.set_font("Arial", size=8)

    if "Metric" not in table_data:
        return None

    columns = list(table_data.keys())
    column_widths = [20] + [20] * (len(columns) - 1)
    table_width = sum(column_widths)
    start_x = (210 - table_width) / 2  # Center table
    
    # Print Table Headers
    pdf.set_x(start_x)
    for i, col in enumerate(columns):
        pdf.cell(column_widths[i], 6, col, border=1, align='C')
    pdf.ln()
    
    # Print Table Rows
    for i, metric in enumerate(table_data["Metric"]):
        pdf.set_x(start_x)
        pdf.cell(column_widths[0], 6, metric, border=1)
        for j, col in enumerate(columns[1:]):
            value = table_data[col][i] if i < len(table_data[col]) else 0
            pdf.cell(column_widths[j + 1], 6, str(round(value, 2)), border=1, align='C')
        pdf.ln()

    # AI Analysis Section
    pdf.ln(10)  # Reduce space
    pdf.set_font("Arial", size=7)
    pdf.multi_cell(0, 6, analysis_text)  # Ensure analysis text appears

    # Save PDF to BytesIO
    pdf_buffer = BytesIO()
    pdf_content = pdf.output(dest="S").encode("latin1")
    pdf_buffer.write(pdf_content)
    pdf_buffer.seek(0)
    
    return pdf_buffer

# --- Streamlit App ---
if "analysis_text" not in st.session_state:
    st.session_state.analysis_text = "No AI analysis made yet."

if "pdf_buffer" not in st.session_state:
    st.session_state.pdf_buffer = None

# User can modify the analysis before generating the PDF
st.session_state.analysis_text = st.text_area("AI Performance Analysis", st.session_state.analysis_text, height=250)

if st.button("Generate AI Analysis"):
    response = client.chat.completions.create(
    model="gpt-4o-mini",  #gpt-3.5-turbo
    messages=[{"role": "user", "content": prompt}]
)

    print(response.choices[0].message.content)

    st.session_state.analysis_text = response.choices[0].message.content


if st.button("Generate PDF"):
    st.session_state.pdf_buffer = generate_pdf([fig1, fig2, fig3], table_data, st.session_state.analysis_text)

# Download Button (Only Show if PDF is Generated)
if st.session_state.pdf_buffer:
    st.download_button(
        label="Download PDF Report",
        data=st.session_state.pdf_buffer,
        file_name="Metabolic_Profile_Report.pdf",
        mime="application/pdf"
    )
