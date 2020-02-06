# Data Science Portfolio
Repository containing portfolio of data science projects completed by me for academic, self learning, and hobby purposes.

   1. Package and Algortithms : Scripting both of these are a part of my course from STAT department : STAT 689 - Statistical Computation. The method which is converted to package is inspired by my research work on industrial projects

   2. Projects : Industial projects along with course and hobby are mentioned. The projects under industrial will be demonstrated with report only, keeping in mind the confidentiality of the code. 

   3. Online course and books : This section highlights the material, both in the form of tutorial and books, which are pivotal for widening my understanding about data science.

   4. Contact : I can be reached out at the given contacts, to talk more about my work, and how I can add value to organizations vision.

## Contents

- ### Package

    - [Package WindPlus](https://github.com/nitesh-1507/WindPlus): The statistical method covariate matching is rooted in deep literature. My current work as a Graduate Research Assistant on Industrial projects, paved way to script the method into a package. The code execution time has been reduced from 30 minutes to 20 seconds using C++.
    
- ### Algorithms from scratch

     - [Linear Regression](https://github.com/nitesh-1507/Linear-Regression): The implementation of linear regression from scratch has been highlighted, which includes gradient calculation and parameter update.
     
     - [Clustering Kmeans](https://github.com/nitesh-1507/Clustering-Kmeans): The vectorized implementation of Kmean clustering has been highlighted. The algorithm showcases the fundamentals of centroid initialization, and its update as the iteration proceeds.
    
    - [Logistic Regression Multiclass](https://github.com/nitesh-1507/Logistic-Regression-Multiclass): The implementation of multiclass logistic regression is highlighted, using newtons method. Unlike gradient descent, newtons method requires computing hessian matrix. The ridge penalty is further added in the objective function to make the inverse of hessian practical. 
     
     - [Lasso using coordinate descent](https://github.com/nitesh-1507/Lasso-Regession-using-coordinate-descent): The implementation of lasso regression, or L1 penalty for sparsity is highlighted. The gradient update is done using coordinate descent, which unlike gradient descent updates one coordinate at a time. 
     
     - [Lasso using C++](https://github.com/nitesh-1507/Lasso-Regression-using-Cpp): The lasso implementation using c++ in R shows significant speed improvement. The armadillo library eases the code implementation in C++, since it consists of easy operation using vectors and matrix. 


- ### Projects
 
   #### Industry Projects

     - [Power curve modeling](https://github.com/nitesh-1507/Power-curve-modelling-and-performance-evaluation-of-wind-energy-data): The methods used to model the wind energy data are KNN, Splines, Gaussian Process, Kernel Regression. Since the data set was that of time series, various resolutions were considered and compared, such as 10 minutes average, 1 minute average, downsampling etc. The RMSE obtained by above methods outperformed industry accepted methodolgy - Binning.

     - [Feature Engineering using Xgboost](https://github.com/nitesh-1507/Data-mining-and-feature-engineering-of-wind-energy-data-using-Xgboost): In normal practices, not every information is important, similarly each of the features does not add to the story telling. Initial filtering of features were done using correlation plot, and histogram similarity analysis. Doing so, reduced the numbr of features from 190 to approximately 35. Further Xgboost was applied to extract the important fetures using importance value, which reduced the number to 10 features. Also new features like standard deviation and turbulence intensity were engineered.

     - [Spatial and Temporal effect on Turbine performance](https://github.com/nitesh-1507/Spatial-and-Temporal-effect-on-turbine-performance): Space and time play a key role in quantifying performance comparison, since the environmental conditions changes with respect to both. The performance quantification was carried out for entire farm, keeping one of the turbine as a reference. The methods used to accomplish are covariate matching, hypothesis testing using gaussian process.
 
   #### Course Projects

     - [Classifying risk of an insurance claim ](https://github.com/nitesh-1507/Classifying-the-risk-of-an-insurance-claim): Defaulters pose a high risk to any financial organization. The data set at hand is quite skewed, out of 100,000 observations only 720 insurances are claimed. This claims cost organization immensely. Methods like Logistic regression, Linear discriminant analysis, Quadratic discriminant analysis are used. Unlike these methods, Anamoly detection or Random under sampling are more promising methods to tackle a situation of skewed data set.

     - [Detection of anomalous points of multivariate data using Principal Component Analysis(PCA)](https://github.com/nitesh-1507/Detection-of-anomalous-points-of-multivariate-data-using-Principal-Component-Analysis-PCA): Multivariate data shows a characteristic of some correlation between features. This property makes it difficult to separate out of control data observations while preparing control limit in manufacturing industry, using schewart chart. Using PCA 209 features were reduced to 10 eigen values, using which projected data were studied individually to separate in control data from out of control one.

- ### Online Courses

     - [Machine Learning Foundations](https://www.coursera.org/learn/ml-foundations)
     - [Machine Learning Regression](https://www.coursera.org/learn/ml-regression)
     - [Machine Learning Classification](https://www.coursera.org/learn/ml-classification)
     - [Machine Learning Clustering](https://www.coursera.org/learn/ml-clustering-and-retrieval)
     - [Neural networks and Deep learning](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning)
     - [Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning)
     - [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects?specialization=deep-learning)
     - [Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks?specialization=deep-learning)
     - [Sequence Models](https://www.coursera.org/learn/nlp-sequence-models)
     - [Tableau 10 A-Z: Hands-On Tableau Training For Data Science!](https://www.udemy.com/course/tableau10/)
   
- ### Books and videos
   
   This is a selection of books and videos for data science and related disciplines that I often use as a good reference
      
     - [Introduction to statistical learning](https://www.amazon.com/Introduction-Statistical-Learning-Applications-Statistics/dp/1461471370)
     - [Element of statistical learning](https://www.amazon.com/Elements-Statistical-Learning-Prediction-Statistics/dp/0387848576)
     - [Convex optimization](https://www.amazon.com/Convex-Optimization-Corrections-2008-Stephen/dp/0521833787/ref=sr_1_1?keywords=Convex+Optimization+By+Stephen+Boyd&qid=1576600464&s=books&sr=1-1)
     - [Introduction to linear algebra](https://www.amazon.com/Introduction-Linear-Algebra-Gilbert-Strang/dp/0980232775/ref=asc_df_0980232775/?tag=hyprod-20&linkCode=df0&hvadid=312152840806&hvpos=1o2&hvnetw=g&hvrand=9003032695276695227&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9027902&hvtargid=pla-454800779501&psc=1&tag=&ref=&adgrpid=61316181319&hvpone=&hvptwo=&hvadid=312152840806&hvpos=1o2&hvnetw=g&hvrand=9003032695276695227&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9027902&hvtargid=pla-454800779501)
     - [Introduction to linear algebra - video](https://www.youtube.com/watch?v=J7DzL2_Na80&list=PLE7DDD91010BC51F8&index=2)
     - [Convex optimization - video](https://www.youtube.com/watch?v=McLq1hEq3UY&list=PL3940DD956CDF0622)
 
- ### Contact
    

     - [LinkedIn](https://www.linkedin.com/in/niteshkumar92/)
     - Email : nitesh.kumar@tamu.edu
     - Phone : (979)-571-8325
