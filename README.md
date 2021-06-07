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
    |    ├── FL 3 Clients 		<- code for the three clients FL scenario (non-iid)
    |           ├── Classification_Performance_3_Clients_non_iid.ipynb	<- plot classification performance for all clients and global model
    |           ├── Explanation_Performance_FL_3_clients_non_iid.ipynb		<- generate heatmaps via SHAP value for all saved models
    |           ├── FL_3_clients_non_iid.ipynb		<- train models in the FL 3 Clients scenario
    
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

Code to plot the results
```
Plotting results
    ├── plot_results						
    |    ├── model
    |           ├── loss_plot.py		<- src for plotting Figure 2
    |           ├── auroc_plot.py		<- src for plotting Figure 3
    |           ├── pred.npy			<- stores predicted labels of trained model on validation set
    |           ├── true.npy			<- stores true labels of validation set

    |    ├── runtime_plot.py			<- src for plotting Figure 4
    
    |    ├── point_removal
    |           ├── point_removal_plot.py	<- src for plotting Figure 6
    |           ├── val_result_HtoL.npz		<- stores results from applications/point_removal/point_removal.py
    
    |    ├── label_detection					
    |           ├── label_plot.py		<- src for plotting Figure 7				
    |           ├── f_knn.pkl			<- stores information about fraction of incorrect labels detected
    |           ├── f_random.pkl		<- stores information about fraction of incorrect labels detected
    |           ├── x_knn.pkl			<- stores information about fraction of data inspected
    |           ├── x_random.pkl		<- stores information about fraction of data inspected
```

--------------------------------------------------------------------------------------------------------------------------------------------------
