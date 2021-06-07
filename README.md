Project Organization
--------------------------------------------------------------------------------------------------------------------------------------------------

Description of the folder structure with all data
```
Final Material
    ├── Code
    |    ├── Benchmark
    |           ├── BenchmarkNet.ipynb			<- train BenchmarkNet
    |           ├── Explanation_Performance_BenchmarkNet.ipynb		<- apply SHAP value for all saved models 
    |    ├── confusion_matrices_roc_curves.ipynb			<- plots confusion matrices and roc curves for the desired models
    |    ├── Data and Preprocessing
    |           ├── CheXpert-v1.0-small			<- folder which would contain the whole CheXpert dataset before data sampling 
    |           ├── create_dataframe.ipynb		<- creates a dataframe containing  
    |           ├── BenchmarkNet.ipynb			<- train BenchmarkNet
    |           ├── Explanation_Performance_BenchmarkNet.ipynb		<- apply SHAP value for all saved models 
    |           ├── Explanation_Performance_BenchmarkNet.ipynb		<- apply SHAP value for all saved models 

    |    ├── point_removal
    |           ├── knn_shap_calculation.py	<- src for calculating KNN-Shapley values
    |           ├── plot_densenet.py		<- src for training and evaluating model after points are removed
    |           ├── point_removal.py		<- src for point removal experiment
    |           ├── utils.py			<- src for utils of knnn_shap_calculation.py and plot_densenet.py
    
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
