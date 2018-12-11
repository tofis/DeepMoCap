using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;

using MahApps.Metro.Controls;
using Ookii.Dialogs.Wpf;
using System.Globalization;

using System.IO;
using Newtonsoft.Json;
using VCL.Moto.Enums;

using System.ComponentModel;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;

namespace MCUDART_Annotator
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : MetroWindow, INotifyPropertyChanged
    {
        private int[][] MARKER_PAIRS = new int[][]
        {
            new int[] {1, 2},
            new int[] {1, 3},
            new int[] {2, 10},
            new int[] { 10, 11},
            new int[] { 11, 12},
            new int[] { 12, 13},
            new int[] { 3, 15},
            new int[] { 15, 16},
            new int[] { 16, 17},
            new int[] { 17, 18},
            new int[] { 1, 19 },
            new int[] { 19, 20},
            new int[] { 20, 21},
            new int[] { 21, 22},
            new int[] { 1, 23},
            new int[] { 23, 24},
            new int[] { 24, 25},
            new int[] { 25, 26},
            new int[] { 4, 6},
            new int[] { 5, 6},
            new int[] { 6, 7},
            new int[] { 7, 8},
            new int[] { 8, 19 },
            new int[] { 8, 23},
            new int[] { 7, 9},
            new int[] { 9, 11},
            new int[] { 7, 14},
            new int[] { 14, 16}
        };

        private JSONPoseFormatDataSet jsonDataset = new JSONPoseFormatDataSet();
        private int IDs = 0;
        private List<Point> _segments = new List<Point>();
        private List<Point> _segments_rgb = new List<Point>();
        private bool _selectSegment = false;
        private int ir_width = 0, ir_height = 0;
        private string ACCEPTED = "ACCEPTED";
        private const int image_width = 512;
        private const int image_height = 424;

        /// <summary>
        /// Window Initialization
        /// </summary>
        public MainWindow()
        {
            jsonDataset.info = new Info();
            jsonDataset.info.contributor = "Anargyros Chatzitofis";
            jsonDataset.info.date_created = DateTime.Now.ToString("yyyy-MM-ddTHH:mm:ss.fff");
            jsonDataset.info.description = "MCUDART Training Dataset";
            jsonDataset.info.version = "1.0.0";
            jsonDataset.info.year = 2018;

            jsonDataset.categories = new List<Category>();
            string cat_json = "{\"supercategory\": \"person\",\"id\": 1,\"name\": \"person\",\"keypoints\":" +
                    "[\"head\", \"F_SPINEMID\", \"F_R_CHEST\", \"F_L_CHEST\", \"F_R_HEAD\",\"F_L_HEAD\", \"B_HEAD\", " +
                    "\"B_BACK\", \"B_SPINEMID\",  \"B_R_SHOULDER\", \"F_R_SHOULDER\", \"R_ELBOW\", " +
                    "\"R_WRIST\", \"R_HAND\", \"B_L_SHOULDER\", \"F_L_SHOULDER\", \"L_ELBOW\", \"L_WRIST\", \"L_HAND\",  " +
                    "\"R_PELVIS\", \"R_CALF\", \"R_ANKLE\", \"R_FOOT\", \"L_PELVIS\", \"L_CALF\", \"L_ANKLE\", \"L_FOOT\"]," +
                    "\"skeleton\": [] }";

            Category category = JsonConvert.DeserializeObject<Category>(cat_json);
            jsonDataset.categories.Add(category);

            jsonDataset.licenses = new List<License>();
            License licence = new License();
            licence.id = 0;
            licence.name = "VCL-NTUA-UCY-TS";
            licence.url = "-";
            jsonDataset.licenses.Add(licence);

            jsonDataset.images = new List<Image>();
            jsonDataset.annotations = new List<Annotation>();

            InitializeComponent();

            if (_selectSegment)
            {
                this.btn_Select.Content = "CORNER";
                this.btn_Select.Background = new SolidColorBrush(Color.FromRgb(0, 0, 0));
            }
            else
            {
                this.btn_Select.Content = "MARKER";
                this.btn_Select.Background = new SolidColorBrush(Color.FromRgb(0, 170, 255));
            }


            this.KeyDown += MainWindow_KeyDown;
        }

        private void MainWindow_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.RightCtrl)
            {
                Accept();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_SetPath_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new VistaOpenFileDialog()
            {
                AddExtension = true,
                CheckFileExists = true,
                CheckPathExists = true,
                DefaultExt = "*.png",
                DereferenceLinks = true,
                Filter = "Recordings (.png)|*.png",
                RestoreDirectory = true,
                Multiselect = false,
                Title = "Select a file ...",
            };

            var dialogresult = dialog.ShowDialog();
            if (dialogresult.HasValue && dialogresult.Value)
            {
                this._path = Path.GetDirectoryName(dialog.FileName);

                this._files_cd = Directory.GetFiles(this._path, "*mc_blob*");
                this._files_of = Directory.GetFiles(this._path, "*flow*");

                //// Uncomment if you have infrared / mask / or depth images or change accordinlgly for loading other images
                //this._filesInfrared = Directory.GetFiles(this._path, "*infrared.png");
                //this._filesMasks = Directory.GetFiles(this._path, "*mask.png");
                //this._files_depth = Directory.GetFiles(this._path, "*depth.pgm");
                //List<string> fileMasks = new List<string>();
                //foreach (var file in this._filesMasks)
                //{
                //    if (!file.Contains("RGB"))
                //    {
                //        fileMasks.Add(file);
                //    }
                //}
                //this._filesMasks = fileMasks.ToArray();

                this._markerFilesIR = Directory.GetFiles(this._path, "*IR.txt");
                this._acceptedFiles = new bool[this._files_cd.Length];
            }

            var result = MessageBox.Show("Do you want to load the files for data export? (JSON)", "Confirmation", MessageBoxButton.YesNo, MessageBoxImage.Question);

            if (result == MessageBoxResult.Yes)
            {
                string cat_json = "{\"supercategory\": \"person\",\"id\": 1,\"name\": \"person\",\"keypoints\":" +
                    "[\"head\", \"F_SPINEMID\", \"F_R_CHEST\", \"F_L_CHEST\", \"F_R_HEAD\",\"F_L_HEAD\", \"B_HEAD\", " +
                    "\"B_BACK\", \"B_SPINEMID\",  \"B_R_SHOULDER\", \"F_R_SHOULDER\", \"R_ELBOW\", " +
                    "\"R_WRIST\", \"R_HAND\", \"B_L_SHOULDER\", \"F_L_SHOULDER\", \"L_ELBOW\", \"L_WRIST\", \"L_HAND\",  " +
                    "\"R_PELVIS\", \"R_CALF\", \"R_ANKLE\", \"R_FOOT\", \"L_PELVIS\", \"L_CALF\", \"L_ANKLE\", \"L_FOOT\"]," +
                    "\"skeleton\": [] }";
                Category category = JsonConvert.DeserializeObject<Category>(cat_json);
                jsonDataset.categories[0] = category;

                for (int i = 0; i < this._files_cd.Length; ++i)
                {
                    ShowImage(i, true);
                    LoadMarkers(i, InputType.D_IR);
                    Console.WriteLine(i + "/" + this._files_cd.Length);
                    File.AppendAllText("console.txt", i + "/" + this._files_cd.Length + "\n");
                }

                ExportJson();
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_Next_Click(object sender, RoutedEventArgs e)
        {
            if (_files_cd != null && _files_cd.Length > 0)
            {
                if (_frameIndex < _files_cd.Length - 1)
                    _frameIndex++;

                ShowImage(_frameIndex, false);
            }
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btn_Previous_Click(object sender, RoutedEventArgs e)
        {
            if (_files_cd != null && _files_cd.Length > 0)
            {
                if (_frameIndex > 0)
                    _frameIndex--;
                ShowImage(_frameIndex, false);
            }
        }

        private void btn_Export_Json(object sender, RoutedEventArgs e)
        {
            ExportJson();
        }

        private void ExportJson()
        {
            string jsonContent = JsonConvert.SerializeObject(jsonDataset);

            DirectoryInfo info = Directory.CreateDirectory("annotation_" + DateTime.Now.ToString("yyyy-MM-ddTHH--mm-ss"));

            File.WriteAllText(info.ToString() + "/person_annotation_train.json", jsonContent);
        }        

        private void LoadMarkers(int index, InputType type)
        {
            if (type == InputType.D_IR)
            {
                if (_files_cd != null && _files_cd.Length > 0)
                {
                    // Fill image info
                    Image img = new Image();
                    img.coco_url = "";
                    img.date_captured = "";
                    img.file_name = _files_cd[index];
                    img.id = IDs;
                    img.license = 0;
                    img.width = ir_width;
                    img.height = ir_height;

                    string[] values = Path.GetFileName(_files_cd[index]).Split(new char[] { '_' }, StringSplitOptions.RemoveEmptyEntries);


                    if (values.Length == 4)
                    {
                        c_index = int.Parse(values[2].Remove(5));
                    }
                    else
                    {
                        c_index = int.Parse(values[4].Remove(5));
                    }


                    jsonDataset.images.Add(img);

                    // Fill annotation info
                    Annotation annotation = new Annotation();
                    annotation.category_id = 1;
                    annotation.image_id = IDs;
                    annotation.iscrowd = 0;
                    annotation.id = IDs;

                    List<int> segs = new List<int>();
                
                    foreach (var segment in _segments)
                    {
                        segs.Add((int)segment.X);
                        segs.Add((int)segment.Y);
                    }

                    annotation.segmentation = new List<List<int>>();
                    annotation.segmentation.Add(segs);

                    int minX = 10000, maxX = 0, minY = 10000, maxY = 0;
                    foreach (var marker in _markerSkeletonIR)
                    {
                        if (marker.Value.X > maxX)
                            maxX = (int)marker.Value.X;
                        if (marker.Value.X < minX)
                            minX = (int)marker.Value.X;
                        if (marker.Value.Y > maxY)
                            maxY = (int)marker.Value.Y;
                        if (marker.Value.Y < minY)
                            minY = (int)marker.Value.Y;
                    }

                    annotation.area = (maxX - minX) * (maxY - minY);
                    annotation.bbox = new List<int>() { minX, minY, maxX - minX, maxY - minY };
                    annotation.image_id = IDs;                   

                    foreach (MCUDART_MARKERS marker in Enum.GetValues(typeof(MCUDART_MARKERS)))
                    {
                        if (!_markerSkeletonIR.ContainsKey(marker))
                        {
                            _markerSkeletonIR.Add(marker, new Point(0, 0));
                        }
                    }
                 
                    annotation.keypoints = new List<int>() {
                        /// Not including previous positions. Comment if you don't need the previous positions.
                        (int)_markerSkeletonIR[MCUDART_MARKERS.F_SPINEMID].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_SPINEMID].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_SPINEMID].X > 0? 2 : 0,

                        (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_CHEST].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_CHEST].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_CHEST].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_CHEST].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_CHEST].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_CHEST].X > 0? 2 : 0,

                        (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_HEAD].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_HEAD].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_HEAD].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_HEAD].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_HEAD].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_HEAD].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.B_HEAD].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_HEAD].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_HEAD].X > 0? 2 : 0,

                        (int)_markerSkeletonIR[MCUDART_MARKERS.B_BACK].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_BACK].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_BACK].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.B_SPINEMID].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_SPINEMID].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_SPINEMID].X > 0? 2 : 0,

                        (int)_markerSkeletonIR[MCUDART_MARKERS.B_R_SHOULDER].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_R_SHOULDER].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_R_SHOULDER].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_SHOULDER].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_SHOULDER].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_SHOULDER].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.R_WRIST].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_WRIST].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_WRIST].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.R_HAND].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_HAND].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_HAND].X > 0? 2 : 0,

                        (int)_markerSkeletonIR[MCUDART_MARKERS.B_L_SHOULDER].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_L_SHOULDER].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_L_SHOULDER].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_SHOULDER].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_SHOULDER].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_SHOULDER].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.L_WRIST].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_WRIST].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_WRIST].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.L_HAND].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_HAND].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_HAND].X > 0? 2 : 0,

                        (int)_markerSkeletonIR[MCUDART_MARKERS.R_PELVIS].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_PELVIS].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_PELVIS].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.R_CALF].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_CALF].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_CALF].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.R_ANKLE].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_ANKLE].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_ANKLE].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.R_FOOT].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_FOOT].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_FOOT].X > 0? 2 : 0,

                        (int)_markerSkeletonIR[MCUDART_MARKERS.L_PELVIS].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_PELVIS].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_PELVIS].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.L_CALF].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_CALF].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_CALF].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.L_ANKLE].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_ANKLE].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_ANKLE].X > 0? 2 : 0,
                        (int)_markerSkeletonIR[MCUDART_MARKERS.L_FOOT].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_FOOT].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_FOOT].X > 0? 2 : 0
                  
                        /// Including previous positions. Uncomment if you don't need the previous positions.
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.F_SPINEMID].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_SPINEMID].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_SPINEMID].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_SPINEMID].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_SPINEMID].X > 0? 2 : 0,

                        //(int)_markerSkeletonIR[MCUDART_MARKERS.F_R_CHEST].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_CHEST].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_R_CHEST].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_R_CHEST].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_CHEST].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.F_L_CHEST].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_CHEST].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_L_CHEST].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_L_CHEST].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_CHEST].X > 0? 2 : 0,

                        //(int)_markerSkeletonIR[MCUDART_MARKERS.F_R_HEAD].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_HEAD].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_R_HEAD].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_R_HEAD].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_HEAD].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.F_L_HEAD].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_HEAD].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_L_HEAD].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_L_HEAD].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_HEAD].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.B_HEAD].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_HEAD].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_HEAD].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_HEAD].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_HEAD].X > 0? 2 : 0,

                        //(int)_markerSkeletonIR[MCUDART_MARKERS.B_BACK].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_BACK].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_BACK].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_BACK].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_BACK].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.B_SPINEMID].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_SPINEMID].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_SPINEMID].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_SPINEMID].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_SPINEMID].X > 0? 2 : 0,

                        //(int)_markerSkeletonIR[MCUDART_MARKERS.B_R_SHOULDER].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_R_SHOULDER].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_R_SHOULDER].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_R_SHOULDER].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_R_SHOULDER].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.F_R_SHOULDER].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_SHOULDER].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_R_SHOULDER].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_R_SHOULDER].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_R_SHOULDER].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.R_WRIST].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_WRIST].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_WRIST].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_WRIST].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_WRIST].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.R_HAND].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_HAND].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_HAND].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_HAND].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_HAND].X > 0? 2 : 0,

                        //(int)_markerSkeletonIR[MCUDART_MARKERS.B_L_SHOULDER].X, (int)_markerSkeletonIR[MCUDART_MARKERS.B_L_SHOULDER].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_L_SHOULDER].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.B_L_SHOULDER].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.B_L_SHOULDER].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.F_L_SHOULDER].X, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_SHOULDER].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_L_SHOULDER].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.F_L_SHOULDER].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.F_L_SHOULDER].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.L_WRIST].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_WRIST].Y,  (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_WRIST].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_WRIST].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_WRIST].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.L_HAND].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_HAND].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_HAND].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_HAND].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_HAND].X > 0? 2 : 0,

                        //(int)_markerSkeletonIR[MCUDART_MARKERS.R_PELVIS].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_PELVIS].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_PELVIS].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_PELVIS].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_PELVIS].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.R_CALF].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_CALF].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_CALF].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_CALF].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_CALF].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.R_ANKLE].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_ANKLE].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_ANKLE].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_ANKLE].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_ANKLE].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.R_FOOT].X, (int)_markerSkeletonIR[MCUDART_MARKERS.R_FOOT].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_FOOT].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.R_FOOT].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.R_FOOT].X > 0? 2 : 0,

                        //(int)_markerSkeletonIR[MCUDART_MARKERS.L_PELVIS].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_PELVIS].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_PELVIS].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_PELVIS].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_PELVIS].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.L_CALF].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_CALF].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_CALF].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_CALF].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_CALF].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.L_ANKLE].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_ANKLE].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_ANKLE].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_ANKLE].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_ANKLE].X > 0? 2 : 0,
                        //(int)_markerSkeletonIR[MCUDART_MARKERS.L_FOOT].X, (int)_markerSkeletonIR[MCUDART_MARKERS.L_FOOT].Y, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_FOOT].X, (int)_p_markerSkeletonIR[MCUDART_MARKERS.L_FOOT].Y, (int)_markerSkeletonIR[MCUDART_MARKERS.L_FOOT].X > 0? 2 : 0
                };
                    int num = 0;

                    for (var i = 2; i < annotation.keypoints.Count; i += 3)
                    {
                        if (annotation.keypoints[i] != 0)
                            num++;
                    }

                    annotation.num_keypoints = num;
                    
                    jsonDataset.annotations.Add(annotation);

                    p_index = c_index;

                    _acceptedFiles[index] = true;

                    IDs++;
                }
            }
        }

        private void btn_Accept_Click(object sender, RoutedEventArgs e)
        {
            Accept();
        }
        private void Accept()
        {
            if (_files_cd != null && _files_cd.Length > 0 && !_acceptedFiles[_frameIndex])
            {
                if (!Directory.Exists(Path.GetDirectoryName(_files_cd[_frameIndex]) + "\\accepted\\"))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(_files_cd[_frameIndex]) + "\\accepted\\");
                }

                try
                {
                    File.Copy(_files_of[_frameIndex], Path.GetDirectoryName(_files_of[_frameIndex]) + "\\accepted\\" + _files_of[_frameIndex].Replace(Path.GetDirectoryName(_files_of[_frameIndex]), ""));

                }
                catch
                {
                    // there is no flow file at the moment
                }

                //// Uncomment if you have infrared / mask / or depth images or change accordinlgly for loading other images
                //File.Copy(_files_depth[_frameIndex], Path.GetDirectoryName(_files_depth[_frameIndex]) + "\\accepted\\" + _files_depth[_frameIndex].Replace(Path.GetDirectoryName(_files_depth[_frameIndex]), ""));
                //File.Copy(_filesInfrared[_frameIndex], Path.GetDirectoryName(_filesInfrared[_frameIndex]) + "\\accepted\\" + _filesInfrared[_frameIndex].Replace(Path.GetDirectoryName(_filesInfrared[_frameIndex]), ""));
                //File.Copy(_filesMasks[_frameIndex], Path.GetDirectoryName(_filesMasks[_frameIndex]) + "\\accepted\\" + _filesMasks[_frameIndex].Replace(Path.GetDirectoryName(_filesMasks[_frameIndex]), ""));

                File.Copy(_files_cd[_frameIndex], Path.GetDirectoryName(_files_cd[_frameIndex]) + "\\accepted\\" + _files_cd[_frameIndex].Replace(Path.GetDirectoryName(_files_cd[_frameIndex]), ""));
                File.Copy(_markerFilesIR[_frameIndex], Path.GetDirectoryName(_markerFilesIR[_frameIndex]) + "\\accepted\\" + _markerFilesIR[_frameIndex].Replace(Path.GetDirectoryName(_markerFilesIR[_frameIndex]), ""));
                
                _acceptedFiles[_frameIndex] = true;

                ShowImage(_frameIndex, false);

                IDs++;
            }
        }

        private void ShowImage(int index, bool export)
        {
            for (var i = 0; i < _temp_Buttons.Count; ++i)
            {
                ParentGrid.Children.Remove(_temp_Buttons[i]);
            }

            _temp_Buttons.Clear();
            _markerSkeleton.Clear();
            _markerSkeletonIR.Clear();
            _jointSkeleton.Clear();
            _jointSkeletonIR.Clear();
            _segments.Clear();
            _segments_rgb.Clear();

            List<System.Drawing.Point> down_points = new List<System.Drawing.Point>();

            //// Uncomment if you have mask images or change accordinlgly for loading other images
            //Image<Gray, byte> mask = new Image<Gray, byte>(this._filesMasks[index]);
            //ir_width = mask.Width;
            //ir_height = mask.Height;            
            //Emgu.CV.Util.VectorOfVectorOfPoint contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
            //Image<Gray, byte> maskcontour = mask.Convert<Gray, byte>();
            //try
            //{
            //    CvInvoke.FindContours(maskcontour, contours, null, RetrType.Ccomp, ChainApproxMethod.ChainApproxSimple, new System.Drawing.Point());
            //}
            //catch (Exception e)
            //{
            //    Console.WriteLine(e.Message);
            //}
            //if (contours.ToArrayOfArray().Length > 0)
            //{
            //    down_points.Clear();
            //    for (int k = 0; k < contours.ToArrayOfArray()[0].Length; ++k)
            //    {
            //        down_points.Add(contours.ToArrayOfArray()[0][k]);
            //    }
            //    maskcontour.Draw(down_points.ToArray(), new Gray(255), -8, LineType.FourConnected, new System.Drawing.Point(0, 0));
            //}
            //foreach (var pt in down_points)
            //{
            //    _segments.Add(new Point(pt.X, pt.Y));
            //}
            //List<System.Drawing.Point> down_points_rgb = new List<System.Drawing.Point>();           

            TextReader trIR = new StreamReader(_markerFilesIR[index]);
            var linesIR = trIR.ReadToEnd().Split(new char[] { '\n', '\r' }, StringSplitOptions.RemoveEmptyEntries);

            foreach (var line in linesIR)
            {
                var values = line.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                var marker = (MCUDART_MARKERS)int.Parse(values[0]);                
                
                if (!_markerSkeletonIR.ContainsKey(marker))
                {
                    _markerSkeletonIR.Add(marker, new Point((int)float.Parse(values[1]), (int)float.Parse(values[2])));
                }
                else
                {
                    // marker is already contained
                }
                
            }

            if (!export)
            {
                if (!_markerSkeletonIR.ContainsKey(MCUDART_MARKERS.L_WRIST) && _markerSkeletonIR.ContainsKey(MCUDART_MARKERS.L_HAND))
                {
                    _markerSkeletonIR.Add(MCUDART_MARKERS.L_WRIST, _markerSkeletonIR[MCUDART_MARKERS.L_HAND]);
                    _markerSkeletonIR.Remove(MCUDART_MARKERS.L_HAND);
                }

                if (!_markerSkeletonIR.ContainsKey(MCUDART_MARKERS.R_WRIST) && _markerSkeletonIR.ContainsKey(MCUDART_MARKERS.R_HAND))
                {
                    _markerSkeletonIR.Add(MCUDART_MARKERS.R_WRIST, _markerSkeletonIR[MCUDART_MARKERS.R_HAND]);
                    _markerSkeletonIR.Remove(MCUDART_MARKERS.R_HAND);

                }

                if (!_markerSkeletonIR.ContainsKey(MCUDART_MARKERS.L_ANKLE) && _markerSkeletonIR.ContainsKey(MCUDART_MARKERS.L_FOOT))
                {
                    _markerSkeletonIR.Add(MCUDART_MARKERS.L_ANKLE, _markerSkeletonIR[MCUDART_MARKERS.L_FOOT]);
                    _markerSkeletonIR.Remove(MCUDART_MARKERS.L_FOOT);

                }

                if (!_markerSkeletonIR.ContainsKey(MCUDART_MARKERS.R_ANKLE) && _markerSkeletonIR.ContainsKey(MCUDART_MARKERS.R_FOOT))
                {
                    _markerSkeletonIR.Add(MCUDART_MARKERS.R_ANKLE, _markerSkeletonIR[MCUDART_MARKERS.R_FOOT]);
                    _markerSkeletonIR.Remove(MCUDART_MARKERS.R_FOOT);
                }

                try
                {

                    if (Point.Subtract(_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW], _markerSkeletonIR[MCUDART_MARKERS.L_HAND]).Length <
                        Point.Subtract(_markerSkeletonIR[MCUDART_MARKERS.L_ELBOW], _markerSkeletonIR[MCUDART_MARKERS.L_WRIST]).Length)
                    {
                        Point temp = _markerSkeletonIR[MCUDART_MARKERS.L_HAND];
                        _markerSkeletonIR[MCUDART_MARKERS.L_HAND] = _markerSkeletonIR[MCUDART_MARKERS.L_WRIST];
                        _markerSkeletonIR[MCUDART_MARKERS.L_WRIST] = temp;
                    }

                }
                catch { }

                try
                {

                    if (Point.Subtract(_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW], _markerSkeletonIR[MCUDART_MARKERS.R_HAND]).Length <
                        Point.Subtract(_markerSkeletonIR[MCUDART_MARKERS.R_ELBOW], _markerSkeletonIR[MCUDART_MARKERS.R_WRIST]).Length)
                    {
                        Point temp = _markerSkeletonIR[MCUDART_MARKERS.R_HAND];
                        _markerSkeletonIR[MCUDART_MARKERS.R_HAND] = _markerSkeletonIR[MCUDART_MARKERS.R_WRIST];
                        _markerSkeletonIR[MCUDART_MARKERS.R_WRIST] = temp;
                    }

                }
                catch { }

                try
                {

                    if (Point.Subtract(_markerSkeletonIR[MCUDART_MARKERS.L_CALF], _markerSkeletonIR[MCUDART_MARKERS.L_FOOT]).Length <
                        Point.Subtract(_markerSkeletonIR[MCUDART_MARKERS.L_CALF], _markerSkeletonIR[MCUDART_MARKERS.L_ANKLE]).Length)
                    {
                        Point temp = _markerSkeletonIR[MCUDART_MARKERS.L_FOOT];
                        _markerSkeletonIR[MCUDART_MARKERS.L_FOOT] = _markerSkeletonIR[MCUDART_MARKERS.L_ANKLE];
                        _markerSkeletonIR[MCUDART_MARKERS.L_ANKLE] = temp;
                    }

                }
                catch { }

                try
                {

                    if (Point.Subtract(_markerSkeletonIR[MCUDART_MARKERS.R_CALF], _markerSkeletonIR[MCUDART_MARKERS.R_FOOT]).Length <
                        Point.Subtract(_markerSkeletonIR[MCUDART_MARKERS.R_CALF], _markerSkeletonIR[MCUDART_MARKERS.R_ANKLE]).Length)
                    {
                        Point temp = _markerSkeletonIR[MCUDART_MARKERS.R_FOOT];
                        _markerSkeletonIR[MCUDART_MARKERS.R_FOOT] = _markerSkeletonIR[MCUDART_MARKERS.R_ANKLE];
                        _markerSkeletonIR[MCUDART_MARKERS.R_ANKLE] = temp;
                    }

                }
                catch { }

                trIR.Close();


                if (index < _files_of.Length)
                {
                    // Image preparation
                    BitmapImage img = new BitmapImage();
                    img.BeginInit();
                    img.UriSource = new Uri(_files_of[index]);
                    img.EndInit();
                    Byte[] pixels = new byte[image_width * image_height * 4];
                    img.CopyPixels(pixels, 4 * image_width, 0);

                    BitmapSource bitmapSource = BitmapSource.Create(image_width, image_height, 96, 96, PixelFormats.Bgra32, null, pixels, 4 * image_width);
                    var visual = new DrawingVisual();
                    using (DrawingContext drawingContext = visual.RenderOpen())
                    {
                        drawingContext.DrawImage(bitmapSource, new Rect(0, 0, image_width, image_height));

                        foreach (var joint in _jointSkeletonIR)
                            drawingContext.DrawEllipse(Brushes.Blue, new Pen(Brushes.Green, 2), joint.Value, 2, 2);


                        foreach (var markerJoint in _markerSkeletonIR)
                        {
                            drawingContext.DrawEllipse(Brushes.Blue, new Pen(Brushes.Blue, 2), markerJoint.Value, 2, 2);
                        }

                        try
                        {
                            for (int i = 0; i < MARKER_PAIRS.Length; ++i)
                            {
                                if (_markerSkeletonIR.ContainsKey((MCUDART_MARKERS)MARKER_PAIRS[i][0]) && _markerSkeletonIR.ContainsKey((MCUDART_MARKERS)MARKER_PAIRS[i][1]))
                                {
                                    drawingContext.DrawLine(new Pen(Brushes.Red, 2), _markerSkeletonIR[(MCUDART_MARKERS)MARKER_PAIRS[i][0]], _markerSkeletonIR[(MCUDART_MARKERS)MARKER_PAIRS[i][1]]);
                                }
                            }
                        }
                        catch (Exception e)
                        {
                            Console.WriteLine(e.Message);
                        }



                        // Create the initial formatted text string.
                        FormattedText formattedText = new FormattedText(
                            ACCEPTED,
                            CultureInfo.GetCultureInfo("en-us"),
                            FlowDirection.LeftToRight,
                            new Typeface("Verdana"),
                            32,
                            Brushes.Black);

                        // Set a maximum width and height. If the text overflows these values, an ellipsis "..." appears.
                        formattedText.MaxTextWidth = 300;
                        formattedText.MaxTextHeight = 240;

                        // Use a larger font size beginning at the first (zero-based) character and continuing for 5 characters.
                        // The font size is calculated in terms of points -- not as device-independent pixels.
                        formattedText.SetFontSize(36 * (96.0 / 72.0), 0, ACCEPTED.Length);

                        // Use a Bold font weight beginning at the 6th character and continuing for 11 characters.
                        formattedText.SetFontWeight(FontWeights.Bold, 0, ACCEPTED.Length);

                        // Use a linear gradient brush beginning at the 6th character and continuing for 11 characters.
                        formattedText.SetForegroundBrush(
                                                new LinearGradientBrush(
                                                Colors.Orange,
                                                Colors.Orange,
                                                90.0),
                                                0, ACCEPTED.Length);

                        // Use an Italic font style beginning at the 28th character and continuing for 28 characters.
                        formattedText.SetFontStyle(FontStyles.Italic, 0, ACCEPTED.Length);

                        if (this._acceptedFiles[index])
                            drawingContext.DrawText(formattedText, new Point(15, 10));
                    }

                    var image = new DrawingImage(visual.Drawing);
                    this.img_Image.Source = image;
                }

                // Image preparation
                BitmapImage imgIR = new BitmapImage();
                imgIR.BeginInit();
                imgIR.UriSource = new Uri(_files_cd[index]);
                imgIR.EndInit();
                Byte[] pixelsIR = new byte[image_width * image_height * 4];
                imgIR.CopyPixels(pixelsIR, 4 * image_width, 0);

                BitmapSource bitmapSourceIR = BitmapSource.Create(image_width, image_height, 96, 96, PixelFormats.Bgra32, null, pixelsIR, 4 * image_width);
                var visualIR = new DrawingVisual();


                using (DrawingContext drawingContext = visualIR.RenderOpen())
                {
                    drawingContext.DrawImage(bitmapSourceIR, new Rect(0, 0, image_width, image_height));

                    try
                    {
                        for (int i = 0; i < MARKER_PAIRS.Length; ++i)
                        {
                            if (_markerSkeletonIR.ContainsKey((MCUDART_MARKERS)MARKER_PAIRS[i][0]) && _markerSkeletonIR.ContainsKey((MCUDART_MARKERS)MARKER_PAIRS[i][1]))
                            {
                                drawingContext.DrawLine(new Pen(Brushes.Green, 3), _markerSkeletonIR[(MCUDART_MARKERS)MARKER_PAIRS[i][0]], _markerSkeletonIR[(MCUDART_MARKERS)MARKER_PAIRS[i][1]]);
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }

                }
                var imageIR = new DrawingImage(visualIR.Drawing);
                img_Image2_mc.Source = new DrawingImage(visualIR.Drawing);

                using (DrawingContext drawingContext = visualIR.RenderOpen())
                {
                    drawingContext.DrawImage(bitmapSourceIR, new Rect(0, 0, image_width, image_height));

                    foreach (var markerJoint in _markerSkeletonIR)
                    {
                        drawingContext.DrawEllipse(new SolidColorBrush(Color.FromArgb(100, 0, 255, 250)), new Pen(Brushes.Magenta, 1), markerJoint.Value, 8, 8);

                        Button bt = new Button();
                        bt.Content = (int)markerJoint.Key;
                        bt.Name = markerJoint.Key.ToString();

                        ParentGrid.Children.Add(bt);
                        Grid.SetColumn(bt, 3);
                        Grid.SetRow(bt, 1);
                        bt.Width = 20;
                        bt.Height = 20;
                        bt.Margin = new Thickness( //markerJoint.Value.X * this.btn_IR.Height / 512, 0, 0, 0);
                            markerJoint.Value.X * this.btn_IR.Height / image_height,
                            markerJoint.Value.Y * this.btn_IR.Height / image_height,
                            (this.btn_IR.Height * image_width / image_height) - markerJoint.Value.X * (this.btn_IR.Height * image_width / image_height) / image_width - bt.Width,
                            this.btn_IR.Height - markerJoint.Value.Y * this.btn_IR.Height / image_height - bt.Height);
                        
                        bt.MouseRightButtonDown += Bt_MouseRightButtonDown;
                        bt.Click += Bt_Click;

                        _temp_Buttons.Add(bt);
                    }

                    try
                    {

                        for (int i = 1; i < _segments.Count; ++i)
                        {
                            drawingContext.DrawLine(new Pen(Brushes.Yellow, 2), _segments[i - 1], _segments[i]);
                        }

                        for (int i = 0; i < MARKER_PAIRS.Length; ++i)
                        {
                            if (_markerSkeletonIR.ContainsKey((MCUDART_MARKERS)MARKER_PAIRS[i][0]) && _markerSkeletonIR.ContainsKey((MCUDART_MARKERS)MARKER_PAIRS[i][1]))
                            {
                                drawingContext.DrawLine(new Pen(Brushes.Red, 2), _markerSkeletonIR[(MCUDART_MARKERS)MARKER_PAIRS[i][0]], _markerSkeletonIR[(MCUDART_MARKERS)MARKER_PAIRS[i][1]]);
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        Console.WriteLine(e.Message);
                    }


                    // Create the initial formatted text string.
                    FormattedText formattedText = new FormattedText(
                        ACCEPTED,
                        CultureInfo.GetCultureInfo("en-us"),
                        FlowDirection.LeftToRight,
                        new Typeface("Verdana"),
                        32,
                        Brushes.Black);

                    // Set a maximum width and height. If the text overflows these values, an ellipsis "..." appears.
                    formattedText.MaxTextWidth = 300;
                    formattedText.MaxTextHeight = 240;

                    // Use a larger font size beginning at the first (zero-based) character and continuing for 5 characters.
                    // The font size is calculated in terms of points -- not as device-independent pixels.
                    formattedText.SetFontSize(36 * (96.0 / 72.0), 0, ACCEPTED.Length);

                    // Use a Bold font weight beginning at the 6th character and continuing for 11 characters.
                    formattedText.SetFontWeight(FontWeights.Bold, 0, ACCEPTED.Length);

                    // Use a linear gradient brush beginning at the 6th character and continuing for 11 characters.
                    formattedText.SetForegroundBrush(
                                            new LinearGradientBrush(
                                            Colors.Orange,
                                            Colors.Orange,
                                            90.0),
                                            0, ACCEPTED.Length);

                    // Use an Italic font style beginning at the 28th character and continuing for 28 characters.
                    formattedText.SetFontStyle(FontStyles.Italic, 0, ACCEPTED.Length);

                    if (this._acceptedFiles[index])
                        drawingContext.DrawText(formattedText, new Point(15, 10));
                }
                imageIR = new DrawingImage(visualIR.Drawing);


                //// Uncomment if you have infrared images or change accordinlgly for loading other images
                //BitmapImage imgInfrared = new BitmapImage();
                //imgInfrared.BeginInit();
                //imgInfrared.UriSource = new Uri(_filesInfrared[index]);
                //imgInfrared.EndInit();
                //Byte[] pixelsInfrared = new byte[image_width * image_height * 4];
                //imgInfrared.CopyPixels(pixelsInfrared, 4 * image_width, 0);

                //BitmapSource bitmapSourceInfrared = BitmapSource.Create(image_width, image_height, 96, 96, PixelFormats.Bgra32, null, pixelsInfrared, 4 * image_width);
                //var visualInfrared = new DrawingVisual();
                //using (DrawingContext drawingContext = visualInfrared.RenderOpen())
                //{
                //    drawingContext.DrawImage(bitmapSourceInfrared, new Rect(0, 0, image_width, image_height));
                //}
                //var imageInfrared = new DrawingImage(visualInfrared.Drawing);

                //this.img_Image2.Source = imageInfrared;

                this.img_ImageIR.Source = imageIR;

                this.lbl_Filename.Content = _files_cd[index].Replace(Path.GetDirectoryName(_files_cd[index]), "");
                this.lbl_Index.Content = _frameIndex.ToString();
            }
        }

        private void Bt_DragLeave(object sender, DragEventArgs e)
        {
            throw new NotImplementedException();
        }

        private void Bt_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
        {
            var btn = (Button)sender;

            _markerSkeletonIR.Remove((MCUDART_MARKERS)int.Parse(btn.Content.ToString()));
            string newtext = "";
            foreach (var marker in _markerSkeletonIR)
            {
                newtext += (int)marker.Key + " " + marker.Value.X + " " + marker.Value.Y + "\n";
            }

            File.WriteAllText(_markerFilesIR[_frameIndex], newtext);

            ParentGrid.Children.Remove((Button)sender);

            ShowImage(_frameIndex, false);
        }

        private void Bt_Click(object sender, RoutedEventArgs e)
        {
            var btn = (Button)sender;

            EditMarker(btn.Content.ToString(), btn.Margin);

            _markerSkeletonIR.Remove((MCUDART_MARKERS)int.Parse(btn.Content.ToString()));
            string newtext = "";
            foreach (var marker in _markerSkeletonIR)
            {
                newtext += (int)marker.Key + " " + marker.Value.X + " " + marker.Value.Y + "\n";
            }

            File.WriteAllText(_markerFilesIR[_frameIndex], newtext);

            ParentGrid.Children.Remove((Button)sender);

            ShowImage(_frameIndex, false);

            e.Handled = true;
        }

        private void EditMarker(string markerID, Thickness margin)
        {          
            if (_selectSegment)
            {
                var x = (int)((int)margin.Left * 512 / this.img_Image.ActualWidth) + this.img_Image.ActualWidth;
                var y = (int)((int)margin.Top * 424 / this.img_Image.ActualHeight);

                _segments.Add(new Point(x, y));
                ShowImage(_frameIndex, false);
            }
            else
            {
                TextBox tb = new TextBox();
                tb.Text = markerID;        

                ParentGrid.Children.Add(tb);

                Grid.SetColumn(tb, 3);
                Grid.SetRow(tb, 1);
                tb.Width = 20;
                tb.Height = 20;
                tb.Margin = margin;
                tb.Focus();
                tb.TextChanged += Tb_TextChanged;
                tb.KeyDown += Tb_KeyDown;
            }
        }

        private void img_Image_Click(object sender, RoutedEventArgs e)
        {
            var xy = Mouse.GetPosition(this.img_Image);

            if (_selectSegment)
            {
                var x = (int)((int)xy.X * 512 / this.img_Image.ActualWidth) + this.img_Image.ActualWidth;
                var y = (int)((int)xy.Y * 424 / this.img_Image.ActualHeight);

                _segments.Add(new Point(x, y));
                ShowImage(_frameIndex, false);
            }
            else
            {
                TextBox tb = new TextBox();
                tb.Text = "";
                
                ParentGrid.Children.Add(tb);

                Grid.SetColumn(tb, 3);
                Grid.SetRow(tb, 1);
                tb.Width = 20;
                tb.Height = 20;
                tb.Margin = new Thickness((int)xy.X, (int)xy.Y, this.img_ImageIR.ActualWidth - (int)xy.X - tb.Width, this.img_ImageIR.ActualHeight - (int)xy.Y - tb.Height);
                tb.Focus();
                tb.TextChanged += Tb_TextChanged;
                tb.KeyDown += Tb_KeyDown;
            }
        }

        private void img_ImageIR_Click(object sender, RoutedEventArgs e)
        {
            var xy = Mouse.GetPosition(this.img_ImageIR);
            Console.WriteLine(xy.X + " " + xy.Y);

            if (_selectSegment)
            {
                var x = (int)(xy.X * 512 / this.img_ImageIR.ActualWidth);
                var y = (int)(xy.Y * 424 / this.img_ImageIR.ActualHeight);

                _segments.Add(new Point(x, y));
                ShowImage(_frameIndex, false);
            }
            else
            {
                TextBox tb = new TextBox();
                tb.Text = "";                
                ParentGrid.Children.Add(tb);
                Grid.SetColumn(tb, 3);
                Grid.SetRow(tb, 1);
                tb.Width = 20;
                tb.Height = 20;
                tb.Margin = new Thickness(xy.X, xy.Y, this.img_ImageIR.ActualWidth - xy.X - tb.Width, this.img_ImageIR.ActualHeight - xy.Y - tb.Height);
                tb.Focus();
                tb.TextChanged += Tb_TextChanged;
                tb.KeyDown += Tb_KeyDown;
            }
        }


        private void img_ImageIR2_Click(object sender, RoutedEventArgs e)
        {
            var xy = Mouse.GetPosition(this.img_Image2_mc);
            Console.WriteLine(xy.X + " " + xy.Y);

            if (_selectSegment)
            {
                var x = (int)(xy.X * 512 / this.img_ImageIR.ActualWidth);
                var y = (int)(xy.Y * 424 / this.img_ImageIR.ActualHeight);

                _segments.Add(new Point(x, y));
                ShowImage(_frameIndex, false);
            }
            else
            {
                TextBox tb = new TextBox();
                tb.Text = "";

                // tb.Arrange(new Rect(xy, new Point(30, 10)));            

                ParentGrid.Children.Add(tb);

                Grid.SetColumn(tb, 3);
                Grid.SetRow(tb, 1);
                tb.Width = 20;
                tb.Height = 20;
                tb.Margin = new Thickness(xy.X, xy.Y, this.img_ImageIR.ActualWidth - xy.X - tb.Width, this.img_ImageIR.ActualHeight - xy.Y - tb.Height);
                tb.Focus();
                tb.TextChanged += Tb_TextChanged;
                tb.KeyDown += Tb_KeyDown;
            }
        }

        private void Tb_KeyDown(object sender, KeyEventArgs e)
        {
            var tbox = (TextBox)sender;

            if (e.Key == Key.Return)
            {
                if (tbox.Text != "" && !_markerSkeletonIR.ContainsKey((MCUDART_MARKERS)int.Parse(tbox.Text)))
                {
                    var x = (int)(tbox.Margin.Left * 512 / this.img_ImageIR.ActualWidth) - 1;
                    var y = (int)(tbox.Margin.Top * 424 / this.img_ImageIR.ActualHeight) - 1;
                    File.AppendAllText(_markerFilesIR[_frameIndex], "\n" + tbox.Text + " " + x + " " + y);
                    Console.WriteLine(x + " " + y);

                    ParentGrid.Children.Remove(tbox);
                    ShowImage(_frameIndex, false);                  
                }
                else
                {
                    MessageBox.Show("This MARKER " + tbox.Text + " is already set, please remove it at first.", "Confirmation", MessageBoxButton.OK, MessageBoxImage.Information);
                }
            }
        }

        private void Tb_TextChanged(object sender, TextChangedEventArgs e)
        {
            int int_out = 0;
            var tbox = (TextBox)sender;

            if (tbox.Text == "")
            {
                return;
            }
            else if (!char.IsDigit(tbox.Text.Last<char>()))
            {
                tbox.Text = tbox.Text.Remove(tbox.Text.Length - 1);
                return;
            }
            else if (!int.TryParse(tbox.Text, out int_out))
            {
                tbox.Text = "";
                return;
            }
            else if (int.Parse(tbox.Text) > 27 || int.Parse(tbox.Text) < 1)
            {
                tbox.Text = tbox.Text.Remove(tbox.Text.Length - 1);
                return;
            }
        }


        private void btn_Select_Click(object sender, RoutedEventArgs e)
        {
            _selectSegment = !_selectSegment;

            if (_selectSegment)
            {
                this.btn_Select.Content = "CORNER";
                this.btn_Select.Background = new SolidColorBrush(Color.FromRgb(0, 0, 0));
            }
            else
            {
                this.btn_Select.Content = "MARKER";
                this.btn_Select.Background = new SolidColorBrush(Color.FromRgb(0, 170, 255));
            }

        }

        private bool[] _acceptedFiles;
        private int _frameIndex = 0;
        private float _sliderValue = 0;
        private List<Button> _temp_Buttons = new List<Button>();
        private Dictionary<MCUDART_MARKERS, Point> _markerSkeleton = new Dictionary<MCUDART_MARKERS, Point>();
        private string _path;
        private string[] _files_of, _files_cd, _markerFilesIR, _markerFiles;
        //// Uncomment if you have infrared / mask / or depth images or change accordinlgly for loading other images
        //private string[] _files_depth, _filesInfrared, _filesMasks;
        private void Slider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            SliderValue = (float)((Slider)sender).Value;

            _frameIndex = (int)(_markerFilesIR.Length * SliderValue / ((Slider)sender).Maximum);

            ShowImage(_frameIndex, false);
        }

        private int c_index = 0, p_index = 0;

        private Dictionary<MCUDART_MARKERS, Point> _markerSkeletonIR = new Dictionary<MCUDART_MARKERS, Point>();
        private Dictionary<MCUDART_MARKERS, Point> _p_markerSkeletonIR = new Dictionary<MCUDART_MARKERS, Point>();
        private Dictionary<PH_SP_JOINTS, Point> _jointSkeletonIR = new Dictionary<PH_SP_JOINTS, Point>();
        private Dictionary<PH_SP_JOINTS, Point> _jointSkeleton = new Dictionary<PH_SP_JOINTS, Point>();

        public event PropertyChangedEventHandler PropertyChanged;
        private void OnPropertyChanged(string info)
        {
            PropertyChangedEventHandler handler = PropertyChanged;
            if (handler != null)
            {
                handler(this, new PropertyChangedEventArgs(info));
            }
        }
        public float SliderValue
        {
            get
            {
                return _sliderValue;
            }

            set
            {
                _sliderValue = value;
                OnPropertyChanged("SliderValue");
            }
        }

        private enum InputType
        {
            RGB,
            IR,
            DEPTH,
            D_IR
        }
    }
}
