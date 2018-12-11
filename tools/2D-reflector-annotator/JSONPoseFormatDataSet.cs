using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MCUDART_Annotator
{

    public class Info
    {
        public string description { get; set; }
        public string url { get; set; }
        public string version { get; set; }
        public int year { get; set; }
        public string contributor { get; set; }
        public string date_created { get; set; }
    }

    public class License
    {
        public string url { get; set; }
        public int id { get; set; }
        public string name { get; set; }
    }

    public class Image
    {
        public int license { get; set; }
        public string file_name { get; set; }
        public string coco_url { get; set; }
        public int height { get; set; }
        public int width { get; set; }
        public string date_captured { get; set; }
        public string flickr_url { get; set; }
        public int id { get; set; }
    }

   

    public class Annotation
    {
        public List<List<int>> segmentation { get; set; }
        public int num_keypoints { get; set; }
        public int area { get; set; }
        public int iscrowd { get; set; }
        public List<int> keypoints { get; set; }
        public int image_id { get; set; }
        public List<int> bbox { get; set; }
        public int category_id { get; set; }
        public object id { get; set; }
    }

    public class Category
    {
        public string supercategory { get; set; }
        public int id { get; set; }
        public string name { get; set; }
        public List<string> keypoints { get; set; }
        public List<List<int>> skeleton { get; set; }
    }

    public class JSONPoseFormatDataSet
    {
        public Info info { get; set; }
        public List<License> licenses { get; set; }
        public List<Image> images { get; set; }
        public List<Annotation> annotations { get; set; }
        public List<Category> categories { get; set; }
    }

}
