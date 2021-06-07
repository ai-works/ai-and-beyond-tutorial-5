Project Organization
--------------------------------------------------------------------------------------------------------------------------------------------------

Description of the folder structure with all data
```
Final Material
    ├── Code
    |    ├── Benchmark 		<- code for the BenchmarkNet
    |           ├── BenchmarkNet.ipynb			<- train BenchmarkNet
    |           ├── Explanation_Performance_BenchmarkNet.ipynb		<- generate heatmaps via SHAP value for all saved models 
    |    ├── confusion_matrices_roc_curves.ipynb			<- plots confusion matrices and roc curves for the desired models
    |    ├── Data and Preprocessing
    |           ├── CheXpert-v1.0-small			<- folder which would contain the whole CheXpert dataset before data sampling 
    |           ├── create_dataframe.ipynb		<- creates a dataframe containing paths and filenames (e.g. in Colab) to ensure comparability  
    |           ├── final_pneumonia_dataframe.csv			<- the dataframe which was used in thesis, to define which images are for training and which for testing, this makes the results between FL and data-centralized more comparable
    |           ├── move_files.ipynb		<- selects the desired images from CheXpert according to the mentioned selection criteria (Pneumonia, No Pneumonia, frontal view and no uncertainty in labels) and put them in two folders named normal and pneumonia. This step can be done locally before uploading to Google Drive.
    |           ├── normal		<- folder which contains no pneunomia images and will be uploaded to Google Drive 
    |           ├── pneumonia		<- folder which contains pneunomia images and will be uploaded to Google Drive  
    |           ├── trains.csv		<- folder and image structure in the original CheXpert dataset and will be used in move_files.ipynb  
    |    ├── FL 3 Clients 		<- code for the three clients FL scenario (non-iid)
    |           ├── Classification_Performance_3_Clients_non_iid.ipynb	<- plot classification performance for all clients and global model
    |           ├── Explanation_Performance_FL_3_clients_non_iid.ipynb		<- generate heatmaps via SHAP value for all saved models
    |           ├── FL_3_clients_non_iid.ipynb		<- train models in the FL 3 Clients scenario
    |    ├── FL 5 Clients 		<- code for the five clients FL scenario (non-iid)
    |           ├── Classification_Performance_5_Clients_non_iid.ipynb	<- plot classification performance for all clients and global model
    |           ├── Explanation_Performance_FL_5_clients_non_iid.ipynb		<- generate heatmaps via SHAP value for all saved models
    |           ├── FL_5_clients_non_iid.ipynb		<- train models in the FL 3 Clients scenario
    ├── Results
    |    ├── Classification Performance 		<- Folder that contains the images for the classification results used in the thesis
    |    ├── Explainability Performance 		<- Folder that contains the images for the explainability performance results used in the thesis
    |           ├── BenchmarkNet			<- contains the heatmaps of BenchmarkNet used in the thesis
    |                  ├── Appendix	<- contains the images of the Benchmark in the appendix
    |                  ├── Results Section	<- contains the images of the Benchmark in the Results Section
    |           ├── detailed comparison - pneumonia and no pneumonia			<- contains the heatmaps in the detailed comparison section
    |           ├── FL 3 Clients			<- contains the heatmaps of the FL 3 Clients Scenario used in the thesis
    |                  ├── Appendix	<- contains the images of the FL 3 Clients Scenario in the appendix
    |                  ├── Results Section	<- contains the images of the FL 3 Clients Scenario in the Results Section
    |           ├── FL 5 Clients			<- contains the heatmaps of FL 5 Clients Scenario used in the thesis
    |                  ├── Appendix	<- contains the images of the FL 5 Clients Scenario in the appendix
    |                  ├── Results Section	<- contains the images of the FL 5 Clients Scenario in the Results Section   
    |    ├── noisy_label					
    |           ├── flip_arrays					
    |                  ├── deep_features	<- stores the new deep features for flipped data created with get_new_deep.py
    |                  ├── raw_data		<- stores the flipped raw data created with generate_data.py
    |           ├── generate_flip_data.py	<- src for generating and storing flipped data in arrays
    |           ├── get_deep_flip.py		<- src for training, obtaining and storing deep features of flipped data in arrays
    |           ├── knn_shap_calculation_flip.py<- src for calculating KNN-Shapley values for flipped data
    |           ├── label_detection.py		<- src for detecting noisy label experiment
    |           ├── utils.py			<- src for utils of knnn_shap_calculation_flip.py and generate_flip_data.py
```

--------------------------------------------------------------------------------------------------------------------------------------------------
