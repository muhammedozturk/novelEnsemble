# novelEnsemble
An ensemble algorithm developed for defect prediction data sets

This algorithm was developed based on triTraining. triTraining utilizes classification error.
Instead, novelENsemble employs some metrics indicating defectiveness of a software.

The algorithm was established on R and C# codes.

=========================TUNING.CS===================================================
1. Include three functions (TuneHyperAdaBoost(),TuneHyperRandomForest(),TuneHyperGradient())
2. They are used for RegressionAdaBoostLearner,RegressionRandomForestLearner,RegressionSquareLossGradientBoostLearner
3. Returns optimal parameter range

In Tuning.cs, "public string dataPath = Resources.Resource1.xalan_2_4;" determines the data set to be exploited in ensemble learning process.

===================================================================================
 
        public double[] TuneHyperAdaBoost()
        {
            #region Read data

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(dataPath));
            var targetName = "bug";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read classification targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            #endregion

            // metric to minimize
            var metric = new MeanSquaredErrorRegressionMetric();

            // Parameter ranges for the optimizer 
            var paramers = new ParameterBounds[]
            {
                new ParameterBounds(min: 1, max: 100, transform: Transform.Linear), // iterations
                new ParameterBounds(min: 0, max: 0.9, transform: Transform.Linear), // learningrate
                 new ParameterBounds(min: 1, max: 15, transform: Transform.Linear), // maxtreeDepth
                  new ParameterBounds(min: 1, max: 4, transform: Transform.Linear), // minSplitSize
            };

            // create random search optimizer
            var optimizer = new RandomSearchOptimizer(paramers, iterations: 30, runParallel: true);

                     // function to minimize
            Func<double[], OptimizerResult> minimize = p =>
            {
                var cv = new RandomCrossValidation<double>(crossValidationFolds: 5, seed: 42);
                var optlearner = new RegressionAdaBoostLearner(iterations: (int)p[0], learningRate: p[1], maximumTreeDepth: (int)p[2], minimumSplitSize: (int)p[3]);
                var predictions = cv.CrossValidate(optlearner, observations, targets);
                var error = metric.Error(targets, predictions);

                return new OptimizerResult(p, error);
            };

            // run optimizer
            var result = optimizer.OptimizeBest(minimize);
            var bestParameters = result.ParameterSet;
            double[] bestList = new double[5];
            bestList[0] = bestParameters[0]; bestList[1] = bestParameters[1]; bestList[2] = bestParameters[2]; bestList[3] = bestParameters[3];
            return bestList;
        }
        =================================================================
        DEFAULT1.aspx====================================================
           //ENSEMBLE
        public void RegressionEnsembleLearner()
        {
            #region read and split data
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(yeni.dataPath));
            var targetName = "bug";

            // read feature matrix
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // creates training test splitter, 
            // Since this is a regression problem, we use the random training/test set splitter.
            // 30 % of the data is used for the test set. 
            var splitter = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24);

            var trainingTestSplit = splitter.SplitSet(observations, targets);
            var trainSet = trainingTestSplit.TrainingSet;
            var testSet = trainingTestSplit.TestSet;
            #endregion
            var net = new NeuralNet();
            // create the list of learners to include in the ensemble
            var ensembleLearners = new IIndexedLearner<double>[]
                {

                new RegressionAdaBoostLearner(iterations:adaParam1,learningRate:adaParam2,maximumTreeDepth:adaParam3,minimumSplitSize:adaParam4),
                new RegressionRandomForestLearner(randomParam1,randomParam2,randomParam3),
                new RegressionSquareLossGradientBoostLearner(iterations:  gradientParam1, learningRate: gradientParam2,  maximumTreeDepth: gradientParam3,
                    featuresPrSplit: gradientParam4, runParallel: false),
                    //new ClassificationNeuralNetLearner(net, iterations: 10,loss:new AccuracyLoss()),
        };



            // create the ensemble learner
            var learner = new RegressionEnsembleLearner(learners: ensembleLearners);

            // the ensemble learnr combines all the provided learners
            // into a single ensemble model.
            var model = learner.Learn(trainSet.Observations, trainSet.Targets);

            // predict the training and test set.
            var trainPredictions = model.Predict(trainSet.Observations);
            var testPredictions = model.Predict(testSet.Observations);

            // since this is a regression problem we are using square error as metric
            // for evaluating how well the model performs.
            var metric = new MeanSquaredErrorRegressionMetric();
            List<string> configurations = new List<string>();
            var metric2 = new RocAucRegressionMetric(1);
            string sonuc = ""; int uzunluk; int iteration = 20; string error; int j = 0;
            //////////////////////////////////// 
            while (iteration < 200)
            {
                /////////////////////////learner configurations
                // ensembleLearners[2] = new RegressionSquareLossGradientBoostLearner(iterations: iteration, learningRate: 0.028, maximumTreeDepth: 12, subSampleRatio: 0.559, featuresPrSplit: 10, runParallel: false);
                learner = new RegressionEnsembleLearner(learners: ensembleLearners);
                model = learner.Learn(trainSet.Observations, trainSet.Targets);
                trainPredictions = model.Predict(trainSet.Observations);
                testPredictions = model.Predict(testSet.Observations);
                //////////////learner configurations
                configurations.Add("maximumTreeDepth: 15,iterations:  " + iteration + ",learningRate: 0.028,  maximumTreeDepth: 12");
                //Label1.Text += "SONUC=" + metric2.Error(testSet.Targets, testPredictions);

                ///////////////WRİTE ERRORS///////////////////
                for (j = 0; j < testPredictions.Length; j++)
                {

                    error = testSet.Targets[j].ToString() + "\t" + testPredictions[j].ToString();
                    using (StreamWriter sw = File.AppendText(@"C:\Users\maruf\Desktop\SharpEnsemble\SharpEnsemble\Resources\sonucAUC.csv"))
                    {
                        sw.WriteLine(error);
                    }
                }
                //////////////////////////////////////////////

                //csv dosyasına yaz        

                uzunluk = configurations.Count;
                for (int i = 0; i < uzunluk; i++)
                {
                    sonuc += configurations[i].ToString();
                    //File.AppendText(@"C:\mainFrame\makaleler\hyperParameter\hyper2\sharlearningExample\App_GlobalResources\sonuc.csv", sonuc);
                }
                sonuc += "   TestError:" + metric2.Error(testSet.Targets, testPredictions).ToString();
                using (StreamWriter sw = File.AppendText(@"C:\Users\maruf\Desktop\SharpEnsemble\SharpEnsemble\Resources\sonucError.csv"))
                {
                    sw.WriteLine(sonuc);
                }

                iteration += 20;
                sonuc = "";
                configurations.Clear();

            }
            j = 0;

            iteration = 0;
            ///////////////////////////////////////
            var testErrorAUC = metric2.Error(testSet.Targets, testPredictions);
            // measure the error on training and test set.
            var trainError = metric.Error(trainSet.Targets, trainPredictions);
            var testError = metric.Error(testSet.Targets, testPredictions);

            // The ensemble model achieves a lower test error 
            // then any of the individual models:

            // RegressionAdaBoostLearner: 0.4005
            // RegressionRandomForestLearner: 0.4037
            // RegressionSquareLossGradientBoostLearner: 0.3936
            TraceTrainingAndTestError(trainError, testError, trainPredictions, testPredictions);

        }
