Project Organization
--------------------------------------------------------------------------------------------------------------------------------------------------

Description of the folder structure and the respective contents
```
Final Material
    ├── Code 		<- code for the Thesis
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
    ├── Results 		<- contains the images used in the thesis as wel the images which were not used 
    |    ├── Classification Performance 		<- Folder that contains the images for the classification performance results used in the thesis
    |    ├── Complete Results.pptx 		<- PowerPoint that contains the whole explainability evolution of the BenchmarkNet and the FL scenarios
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
    ├── Tex - Master Thesis 		<- contains the .tex files 
```

--------------------------------------------------------------------------------------------------------------------------------------------------
