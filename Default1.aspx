using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using SharpLearning.AdaBoost.Learners;
using SharpLearning.Common.Interfaces;
using SharpLearning.CrossValidation.CrossValidators;
using SharpLearning.CrossValidation.TrainingTestSplitters;
using SharpLearning.DecisionTrees.Learners;
using SharpLearning.Ensemble.Learners;
using SharpLearning.GradientBoost.Learners;
using SharpLearning.InputOutput.Csv;
using SharpLearning.Metrics.Regression;
using SharpLearning.Optimization;
using SharpLearning.RandomForest.Learners;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;
using SharpLearning.Neural.Learners;
using SharpLearning.Neural;
using SharpLearning.Neural.Loss;

namespace SharpEnsemble
{
    public partial class Default1 : System.Web.UI.Page
    {
        public int adaParam1; public double adaParam2; public int adaParam3; public int adaParam4;
        public int gradientParam1; public double gradientParam2; public int gradientParam3; public int gradientParam4;
        public int randomParam1; public int randomParam2; public int randomParam3;
        Tuning yeni;
        protected void Page_Load(object sender, EventArgs e)
        {

            // RegressionEnsembleLearner();
            yeni = new Tuning();
            double[] sonuc1 = yeni.TuneHyperGradient();
            gradientParam1 = (int)sonuc1[0]; gradientParam2 = sonuc1[1]; gradientParam3 = (int)sonuc1[2]; gradientParam4 = (int)sonuc1[3];
            double[] sonuc2 = yeni.TuneHyperAdaBoost();
            adaParam1 = (int)sonuc2[0]; adaParam2 = sonuc2[1]; adaParam3 = (int)sonuc2[2]; adaParam4 = (int)sonuc2[3];
            double[] sonuc3 = yeni.TuneHyperRandomForest();
            randomParam1 = (int)sonuc3[0]; randomParam2 = (int)sonuc3[1]; randomParam3 = (int)sonuc3[2];
            for (int i = 0; i < sonuc1.Length; i++)
                Label1.Text += "/////" + sonuc1[i].ToString();
            for (int i = 0; i < sonuc2.Length; i++)
                Label2.Text += "/////" + sonuc2[i].ToString();
            for (int i = 0; i < sonuc3.Length; i++)
                Label3.Text += "/////" + sonuc3[i].ToString();
            RegressionEnsembleLearner();

        }
        public void RegressionLearner_Learn_And_Predict()
        {
            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(yeni.dataPath));
            var targetName = "bug";

            // read feature matrix (all columns except quality)
            var observations = parser.EnumerateRows(c => c != targetName)
                .ToF64Matrix();

            // read regression targets
            var targets = parser.EnumerateRows(targetName)
                .ToF64Vector();

            // create learner (RandomForest)
            var learner = new RegressionRandomForestLearner(trees: 100);

            // learns a RegressionDecisionTreeModel
            var model = learner.Learn(observations, targets);

            // use the model to predict the training data
            var predictions = model.Predict(observations);
            string myresult = "";
            foreach (var myscore in predictions)
            {

                myresult += myscore.ToString();


            }

            Label1.Text = myresult;
        }
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
        //TUNEHYPER
        public void TuneHyper()
        {
            #region Read data

            // Use StreamReader(filepath) when running from filesystem
            var parser = new CsvParser(() => new StringReader(Resources.Resource1.ant_1_7));
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
                new ParameterBounds(min: 1, max: 1000, transform: Transform.Linear), // iterations
                new ParameterBounds(min: 1, max: 4, transform: Transform.Linear), // learningrate
                 new ParameterBounds(min: 1, max: 15, transform: Transform.Linear), // maxtreeDepth
            };

            // create random search optimizer
            var optimizer = new RandomSearchOptimizer(paramers, iterations: 30, runParallel: true);

            // other availible optimizers
            // GridSearchOptimizer
            // GlobalizedBoundedNelderMeadOptimizer
            // ParticleSwarmOptimizer
            // BayesianOptimizer

            // function to minimize
            Func<double[], OptimizerResult> minimize = p =>
            {
                var cv = new RandomCrossValidation<double>(crossValidationFolds: 5, seed: 42);
                var optlearner = new RegressionRandomForestLearner(trees: (int)p[0], minimumSplitSize: (int)p[1], maximumTreeDepth: (int)p[2]);
                var predictions = cv.CrossValidate(optlearner, observations, targets);
                var error = metric.Error(targets, predictions);

                return new OptimizerResult(p, error);
            };

            // run optimizer
            var result = optimizer.OptimizeBest(minimize);
            var bestParameters = result.ParameterSet;

            Label1.Text = "//////" + bestParameters[0] + "//////" + bestParameters[1] + "//////" + bestParameters[2];


            //var learner = new RegressionDecisionTreeLearner(maximumTreeDepth: (int)bestParameters[0], minimumSplitSize: (int)bestParameters[1]);

            //// learn model with found parameters
            //var model = learner.Learn(observations, targets);

        }

        //FIND RANDOM FOREST PARAMETERS
        //FIND RANDOM FOREST PARAMETERS
        //FIND RANDOM FOREST PARAMETERS
        public double[] randomParameterCompute()
        {
            var parameters = new ParameterBounds[]
         {
    new ParameterBounds(min: 20, max: 300, transform: Transform.Linear), // iterations
    new ParameterBounds(min: 0.02, max:  0.2, transform: Transform.Logarithmic), // learning rate
    new ParameterBounds(min: 8, max: 15, transform: Transform.Linear), // maximumTreeDepth
    new ParameterBounds(min: 0.5, max: 0.9, transform: Transform.Linear), // subSampleRatio
         };
            double[] x = new double[5];
            var metric = new MeanSquaredErrorRegressionMetric();
            var parser = new CsvParser(() => new StringReader(Resources.Resource1.ant_1_7));
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
            var validationSplit = new RandomTrainingTestIndexSplitter<double>(trainingPercentage: 0.7, seed: 24)
    .SplitSet(trainSet.Observations, trainSet.Targets);

            Func<double[], OptimizerResult> minimize = p =>
            {
                p = new double[5];
                p[0] = 100; p[1] = 0.2; p[3] = 0.9; p[4] = 4; p[2] = 15;
                // create the candidate learner using the current optimization parameters.
                RegressionSquareLossGradientBoostLearner candidateLearner = new RegressionSquareLossGradientBoostLearner(iterations: (int)p[0],
                                                                                                                                            learningRate: p[1],
                                                                                                                                            maximumTreeDepth: (int)p[2],
                                                                                                                                            subSampleRatio: p[3],
                                                                                                                                          featuresPrSplit: (int)p[4],
                                                                                                                                            runParallel: false);



                var candidateModel = candidateLearner.Learn(validationSplit.TrainingSet.Observations,
                    validationSplit.TrainingSet.Targets);

                var validationPredictions = candidateModel.Predict(validationSplit.TestSet.Observations);
                var candidateError = metric.Error(validationSplit.TestSet.Targets, validationPredictions);

                return new OptimizerResult(p, candidateError);
            };
            var optimizer = new RandomSearchOptimizer(parameters, iterations: 30, runParallel: true);

            // find best hyperparameters
            var result = optimizer.OptimizeBest(minimize);
            var best = result.ParameterSet;
            x[0] = best[0]; x[1] = best[1]; x[2] = best[2]; x[3] = best[3]; x[4] = best[4];
            Label1.Text = x[0] + "/////" + x[1] + "/////" + x[2] + "/////" + x[3] + "/////" + x[4] + "/////";
            return x;
        }
        public void TraceTrainingAndTestError(double trainError, double testError, double[] x, double[] y)
        {
            //Label1.Text = "Train error: " + trainError + "Test error:" + testError + "train=" + x + "test=" + y;
            //int length1 = y.Length;
            //for (int i = 0; i < length1; i++)
            //    Label1.Text += y[i] + "\n";
        }
    }
}
