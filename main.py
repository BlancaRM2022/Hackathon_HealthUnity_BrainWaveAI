import gradio as gr
import mne
from scipy import signal
import numpy as np
from openai import OpenAI
import joblib
from dotenv import load_dotenv
import os

# Load API address
load_dotenv()
API_KEY = os.getenv('SECRET_API_KEY')


def load_eeg(file_path):  # Step 1: Load EEG File
    # Now Load the File
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    # Retrieve User Information
    data = raw.get_data()
    chn_labels = raw.ch_names
    sample_freq = int(raw.info['sfreq'])

    return data, chn_labels, sample_freq


def load_model(model_path):  # Step 2: Load Trained Model
    learner = joblib.load(model_path)
    return learner


def compute_coherence(main_data, compare_data, sample_freq):  # Step 3A: Function for Calculating Channel Coherence
    # Frequencies we want to extract
    freq_bounds = [13, 20]

    # Channel Storage
    chan_coherence = np.zeros((1, 2))
    # Now loop through all the electrodes and compute the coherence between
    for row_id in range(np.shape(main_data)[0]):
        # Compute the Coherence
        f, cxy = signal.coherence(compare_data, main_data[row_id, :], sample_freq)
        tf_idx = np.logical_and(f >= freq_bounds[0], f <= freq_bounds[1])
        chan_coherence[0, row_id] = np.mean(cxy[tf_idx]) / np.mean(cxy)
    return chan_coherence


def frequency_power(data, sample_freq, time_length):  # Step 3B: Function to Calculate Alpha Power
    # Frequencies we want to extract
    freq_bounds = [8, 12]

    # Frequency Information
    n = time_length * sample_freq
    frequencies = np.linspace(0, sample_freq / 2, n // 2 + 1)
    x = np.fft.fft(data, axis=1)
    power_spectrum = np.abs(x[:, 0:(n // 2 + 1)]) ** 2

    # Identify Frequencies of interest
    tf_num = np.logical_and(frequencies >= freq_bounds[0], frequencies <= freq_bounds[1])
    alpha_power = np.mean(power_spectrum[:, tf_num], axis=1)

    return alpha_power


def categorize_patient(data, chn_labels, sample_freq, learner):  # Step 3C: Coherence and alpha power to predict

    # Extract the Electrodes we want
    fz_data = data[chn_labels.index('Fz'), :]
    model_data = data[chn_labels.index('O1'), :]
    model_data = np.vstack((model_data, data[chn_labels.index('O2'), :]))

    # Check how many potential segments there are
    time_length = 10
    total_segments = (np.shape(model_data)[1]) / time_length
    max_testing = 20
    epoch_classification = [None] * max_testing
    model_statistics = np.zeros((max_testing, 4))

    for idx in range(max_testing):
        # Get The Time Segments
        t1 = (idx * time_length * sample_freq)
        t2 = t1 + time_length * sample_freq

        # Extract the Data From Patients
        temp_fz = fz_data[t1:t2]
        temp_model_data = model_data[:, t1:t2]

        # Extract the Power Markers First
        marker1 = frequency_power(temp_model_data, sample_freq, time_length)
        marker2 = compute_coherence(temp_model_data, temp_fz, sample_freq)

        # Prepare Model Parameters
        model_param = np.hstack((marker1, marker2[0]))
        model_statistics[idx, :] = model_param

        # Let's Predict!
        y_pred = learner.predict(model_param.reshape(1, -1))
        epoch_classification[idx] = y_pred[0]

    # Compute Result Statistics
    percent_control = (epoch_classification.count(0) / max_testing) * 100
    percent_abnormal = (epoch_classification.count(1) / max_testing) * 100
    model_means = np.mean(model_statistics, axis=0)
    model_std = np.std(model_statistics, axis=0)

    return percent_control, percent_abnormal, model_means, model_std


def prompt_answer(percent_control, percent_abnormal, model_means, model_std):  # Step 4: Send predictions to chatbot for interpretation

    # Connect to NVIDIA's API with OpenAI SDK for interpretation
    client = OpenAI(
      base_url="https://integrate.api.nvidia.com/v1",
      api_key=API_KEY
    )

    # Format the prediction result into a prompt
    prompt = (
        f"Task: Analyze EEG Data for Alzheimer's Diagnosis\n\n"
        f"Data Description: We have scalp EEG data from a patient with suspected Alzheimer's disease. "
        f"The analysis was performed using a machine learning classifier focused on **alpha power** and coherence "
        f"of occipital scalp electrodes.\n\n"
        f"Results:\n"
        f"- Classification Outcome: {percent_control}% of EEG epochs classified as **normal**, "
        f"{percent_abnormal}% as **abnormal**.\n"
        f"- Model Statistics:\n"
        f"  - Means: {model_means} (first two values = average alpha power; last two = relative coherence with Fz channel)\n"
        f"  - Standard Deviations: {model_std} (same order as means)\n\n"
        f"**Request:**\n"
        f"1. Interpretation: Provide an explanation of the classification results and determine if it is significant \
        according to the percentage.\n"
        f"2. Clinical Recommendations: Offer guidance for the clinician based on the findings.\n"
        f"3. Biological Explanation: Explain why alpha power and coherence of occipital electrodes are critical in "
        f"diagnosing Alzheimer's disease.\n\n"
        f"Output Format: Present key insights in bullet points."
    )

    # Call NVIDIA API model to interpret
    response = client.chat.completions.create(
      model="writer/palmyra-med-70b-32k",
      messages=[{"role": "user", "content": prompt}],
      temperature=0.2,
      top_p=0.7,
      max_tokens=1024,
    )

    if not response.choices:
        return None
    return response.choices[0].message.content  # model prompt answer


def main(file_path, model_path):  # Main function integrating all steps
    data, chn_labels, sample_freq = load_eeg(file_path)
    learner = load_model(model_path)
    percent_control, percent_abnormal, model_means, model_std = categorize_patient(data, chn_labels, sample_freq, learner)
    return prompt_answer(percent_control, percent_abnormal, model_means, model_std)


def read_the_file(file_path, model_path):
    if file_path is not None:
        print(f"Attempting to read file: {file_path}")
        llm_answer = main(file_path, model_path)
        return llm_answer
    else:
        print("no file was selected")


def read_eeg_get_answer(image_path):
    with gr.Blocks() as demo:
        # Add logo
        gr.Image(image_path)

        # Add explanation paragraph
        gr.Markdown("This tool combines machine learning and advanced language models to analyze scalp EEG data, \
        providing insights into the likelihood of Alzheimer’s disease. It offers a quick, non-invasive assessment, \
        aiming to support early detection and intervention planning.")

        # Layout for file path input and button
        with gr.Row(equal_height=True):
            file_path_input = gr.Textbox(label="File Path", placeholder="Enter the full file path here")
            button = gr.Button("Analyze EEG Data", variant="primary")

        # Button click function
        button.click(read_the_file,
                     inputs=file_path_input,
                     outputs=gr.Textbox(value="", label="Output", lines=10))
    # Launch the demo
    demo.launch()


read_eeg_get_answer()
