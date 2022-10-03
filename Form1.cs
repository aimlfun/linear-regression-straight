using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Security.Cryptography;

namespace Points
{
    /// <summary>
    /// Demonstrates the most rudimentary examples of linear regression (a straight-ish line thru data points).
    /// From posting https://aimlfun.com/linear-regression/
    /// </summary>
    public partial class FormLinearRegression : Form
    {
        /// <summary>
        /// When set to true it saves an image of the neuron output to d:\temp\
        /// </summary>
        private const bool c_saveProgressIntermittently = false;

        /// <summary>
        /// Location of green crosses.
        /// </summary>
        private readonly List<Point> pointsGreenCross = new();

        /// <summary>
        /// Location of red crosses.
        /// </summary>
        private readonly List<Point> pointsRedCross = new();

        /// <summary>
        /// Data containing crosses, to train.
        /// </summary>
        private readonly List<double[]> trainingData = new();

        /// <summary>
        /// Width of the picture box.
        /// </summary>
        private readonly int width;

        /// <summary>
        /// Height of the picture box.
        /// </summary>
        private readonly int height;

        /// <summary>
        /// Epoch is the generation.
        /// </summary>
        private int epoch = 0;

        /// <summary>
        /// Constructor.
        /// </summary>
        public FormLinearRegression()
        {
            InitializeComponent();

            // weirdly access picturebox/image height or width is slow. It's like it has to calculate it.
            height = pictureBoxCrosses.Height;
            width = pictureBoxCrosses.Width;

            // the idea is to put a sea of green crosses, and have red crosses in a small part
            GenerateRandomRedGreenCrosses();
            
            int[] AIHiddenLayers = new int[] { 1, 10, 10, 10, 10, 10, 1 };
            // or do this int[] AIHiddenLayers = new int[] { 1, pointsGreenCross.Count, pointsGreenCross.Count, pointsGreenCross.Count, 1 }; 

            ActivationFunctions[] AIactivationFunctions = new ActivationFunctions[] { ActivationFunctions.TanH, ActivationFunctions.TanH,
                                                                                      ActivationFunctions.TanH, ActivationFunctions.TanH,
                                                                                      ActivationFunctions.TanH, ActivationFunctions.TanH,
                                                                                      ActivationFunctions.TanH};

            NeuralNetwork.s_networks.Clear();

            _ = new NeuralNetwork(0, AIHiddenLayers, AIactivationFunctions);
        }

        /// <summary>
        /// Fills pointsRedCrosses/pointsGreenCrosses with the location of the crosses (indicating 1/0)
        /// </summary>
        private void GenerateRandomRedGreenCrosses()
        {
            // to compare if a point is in the "box" for red crosses, we need to order the top/left-bottom/right coordinates.
            int step;

            int y = height / 2;
            int x = 0;
            float angle = 0;

            // add the crosses
            while (x < width)
            {
                if (RandomNumberGenerator.GetInt32(0, 100) < 20)
                {
                    angle += RandomNumberGenerator.GetInt32(-15, 15);
                    angle = angle.Clamp(-45, 45);
                }

                double angleInRadians = MathUtils.DegreesInRadians(angle);

                step = RandomNumberGenerator.GetInt32(1, 5);
                y += (int)Math.Round(Math.Sin(angleInRadians) * step * 8);
                y = y.Clamp(0, height);
                x += (int)Math.Round(Math.Cos(angleInRadians) * step) + 5;

                pointsGreenCross.Add(new Point(x, y));
            }
        }

        /// <summary>
        /// On load, we make training data out of the crosses. A timer is started to backpropagate.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Form1_Load(object sender, EventArgs e)
        {
            PlotTheCrosses(pictureBoxCrosses);

            // define training data based on points
            foreach (Point p in pointsGreenCross) trainingData.Add(new double[] { (float)p.X / width, (float)p.Y / height});

            timerTrainAndPlot.Tick += TimerTrainAndPlot_Tick;
            timerTrainAndPlot.Start();
        }

        /// <summary>
        /// Plots the red and green crosses.
        /// </summary>
        /// <param name="pb"></param>
        /// <param name="redCrossPoints"></param>
        /// <param name="greenCrossPoints"></param>
        private void PlotTheCrosses(PictureBox pb)
        {
            Bitmap bitmap = new(pb.Width, pb.Height);

            using Graphics g = Graphics.FromImage(bitmap);
        
            g.Clear(Color.Black);
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;

            foreach (Point p in pointsGreenCross) DrawX(g, p, Pens.Lime);
            foreach (Point p in pointsRedCross) DrawX(g, p, Pens.Red);

            g.Flush();

            pb.Image?.Dispose();
            pb.Image = bitmap;
        }

        /// <summary>
        /// Train and plot
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void TimerTrainAndPlot_Tick(object? sender, EventArgs e)
        {
            NeuralNetwork neuralNetwork = NeuralNetwork.s_networks[0];

            labelEpoch.Text = $"Epoch {++epoch}";

            int step = 6;

            Train();

            // if (epoch % 100 != 0) return;

            pointsRedCross.Clear();

            // plot the "predicted"/"learnt" output          
            for (int x = 0; x < width; x += step)
            {
                double[] result = neuralNetwork.FeedForward(new double[] { (float)x / width });

                pointsRedCross.Add(new Point(x, (int)Math.Round(result[0] * height)));
            }
            
            PlotTheCrosses(pictureBoxNeuralNetworkOutput);

            if (c_saveProgressIntermittently) pictureBoxNeuralNetworkOutput.Image.Save($@"c:\temp\linear-regression-epoch-{epoch}.png", ImageFormat.Png);
        }

        /// <summary>
        /// Train the neural network in a random order using training data.
        /// </summary>
        private void Train()
        {
            for (int i = 0; i < trainingData.Count; i++)
            {
                double[] d = trainingData[i];
                NeuralNetwork.s_networks[0].BackPropagate(new double[] { d[0] }, new double[] { d[1] });
            }
        }

        /// <summary>
        /// Draws a red or green "x".
        /// </summary>
        /// <param name="g"></param>
        /// <param name="position"></param>
        /// <param name="p"></param>
        private static void DrawX(Graphics g, Point position, Pen p)
        {
            // x marks the spot for center of mass
            g.DrawLine(p, position.X - 2, position.Y - 2, position.X + 2, position.Y + 2);
            g.DrawLine(p, position.X - 2, position.Y + 2, position.X + 2, position.Y - 2);
        }

        /// <summary>
        /// Enable it to be paused and unpaused.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void FormLinearRegression_KeyDown(object sender, KeyEventArgs e)
        {
            // [P] pause
            if (e.KeyCode == Keys.P) timerTrainAndPlot.Enabled = !timerTrainAndPlot.Enabled;
        }
    }
}