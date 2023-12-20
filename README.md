# kNN

### A comparison of different kNN algorithms written in Python



<a name="readme-top"></a>


<!-- kNN gif -->
![knn](https://github.com/UreshiiPanda/kNN/assets/39992411/aea75699-f792-4027-b9e2-a5b365d5cd81)



<!-- ABOUT THE PROJECT -->
## About The Project

This program offers a comparison of different takes on the kNN algorithm for income classification. It compares how
well the kNN algorithm can classify income data with different data normalization techniques, including: without
binarization, with binarization, and with scaling. The program also implements two versions of kNN which use different
metrics for their distance calculations: Euclidean and Manhattan. In addition, these 2 versions of kNN are also compared to
sklearn's implementation. Finally, each model is run through k-neighbors values from 1 to 99 to compare how the number of 
neighbors affects the results. I worked on this project with 1 other.

#### Results

The overall best error rate on dev was 14.3% and this was achieved at k = 41 using Manhattan distance while also applying
a OneHotEncoder and a MinMaxScaler to normalize and scale the data. The Manhattan distance only provides a marginal 
improvement over the Euclidean diatance metric. When only a OneHotEncoder is applied, the algorithms perform poorly since
the data has a mix of categorical and numerical data, thus all of the numericals get treated as categorical and the results
suffer. The MinMaxScaler helps reduce the dev error from 15.7% to 14.3%. The number of neighbors from 37 to 43 seem to capture
the best fit between overfitting and underfitting in this case.



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This program can be run in a Jupyter Notebook via the steps below.


### Installation / Execution Steps in Jupyter Lab:

1. Clone the repo
   ```sh
      git clone https://github.com/UreshiiPanda/kNN.git
   ```

2. Open the project in a Jupyter Notebook and run each of the cells in graph.ipynb to view
   various results and also plot the results against each other. Note that there are comments
   indicating further directions for comparing 3 different kNN versions.
   ```sh
       Jupyter Lab
   ```

4. Run the following in a shell to check your income predictions against the validator:
   1. 
      ```sh
          python3 predictions.py > income.test.predicted
      ```
   2. 
      ```sh
         cat income.test.predicted | python3 validate.py
      ```
  



<p align="right">(<a href="#readme-top">back to top</a>)</p>

