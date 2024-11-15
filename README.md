**EEG Alzheimer's Detection Using LLM and ML Classifier**

This project is designed to analyze EEG data and predict the likelihood of Alzheimer's disease using a combination of a Large Language Model (LLM) and a Machine Learning (ML) classifier. The tool provides an interface for uploading EEG data and receiving diagnostic predictions.

**Features**

**EEG Data Upload:** Easily upload EEG files for analysis.

**Integrated Analysis:** Utilizes both LLM and ML models to enhance prediction accuracy.

**User Interface:** Built with the "gradio" package for a seamless user experience.

**Integration with OpenAI and NVIDIA:** This project utilizes NVIDIA's [Palmyra-Med-70B-32K](https://build.nvidia.com/writer/palmyra-med-70b-32k) API in collaboration with OpenAI's advanced language models to enhance the analysis and prediction capabilities of the system.

**Installation**

**Clone the Repository:**

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
**Install Dependencies:**

Ensure you have Python installed.
Install the necessary packages using the provided text file:

```bash
pip install -r requirements.txt
```
**Set Up API Key:**

Obtain an API key from [nVIdia](https://build.nvidia.com/explore/discover) by making an account.
Create a .env file in the project folder and add your API key:

```SECRET_API_KEY=your_api_key_here```

**Usage**
**Download EEG Files:** Obtain the EEG files from the "Data" folder in this repository.
s
**Modify the code:** Change the ML Path to your the path on your local PC in the main code script.

**Run the Main Function:** Execute the main script to start the application.

**Access the User Interface:** Upon running the main function, a link will be generated. Follow this link to access the user interface.

![Screenshot 2024-11-14 at 7 16 48 PM](https://github.com/user-attachments/assets/130c9d1a-20d7-4d33-b54f-e21b83d40497)


**Navigate to the Analysis Page:** The link will direct you to the analysis page. 

![Screenshot 2024-11-14 at 7 07 32 PM](https://github.com/user-attachments/assets/d30af847-f060-4223-bc84-ea6228a90bb6)
**Input EEG File Path:** In the "File Path" section at the top, enter the local path to your EEG file. The path should resemble the following format:
downloads\Data\sub-001_task-eyesclosed_eeg.set

**Analyze the Data:** Click the "Analyze EEG Data" button, which is highlighted in orange.

**View the Results:** The output section will display the LLM prompt with the analysis results.



**Contributing**

We welcome contributions from the community. Please feel free to submit issues or pull requests.

**License**

This project is licensed under the MIT License. See the LICENSE file for details.

**Contact**

For questions or support, please contact Soudeh (Venus) Mostaghimi at mostaghs@uci.edu, Trisha Mendoza at trisham@uci.edu, or Blanca Romer at blancr1@uci.edu.
