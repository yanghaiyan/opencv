#include"tld_utils.h"
#include "opencv2/opencv.hpp"

//ʹ�ý�����LK���������٣�������ĳ�Ա�����ܶ඼��OpenCV��calcOpticalFlowPyrLK()�����Ĳ���
class LKTracker{
private:
  std::vector<cv::Point2f> pointsFB;
  cv::Size window_size;  //ÿ������������������ڳߴ�
  int level;            //���Ľ���������
  std::vector<uchar> status;   //���顣�����Ӧ�����Ĺ��������֣������е�ÿһ��Ԫ�ض�������Ϊ 1�� ��������Ϊ 0
  std::vector<uchar> FB_status;   
  std::vector<float> similarity;  //���ƶ�
  std::vector<float> FB_error;   //Forward-Backward error��������FB_error�Ľ����ԭʼλ�õ�ŷʽ����
                                 //���Ƚϣ��Ѿ������ĸ��ٽ������
  float simmed;
  float fbmed;
  //TermCriteriaģ���࣬ȡ����֮ǰ��CvTermCriteria�����������Ϊ�����㷨����ֹ������
  //���������Ҫ3��������һ�������ͣ��ڶ�������Ϊ�����������������һ�����ض�����ֵ��
  //ָ����ÿ���������㣬Ϊĳ��Ѱ�ҹ����ĵ������̵���ֹ������
  cv::TermCriteria term_criteria;
  float lambda;   //ĳ��ֵ����Lagrangian ����
  // NCC ��һ��������أ�FB error��NCC��ϣ�ʹ���ٸ��ȶ�  ������ص�ͼ��ƥ���㷨����
  //������ط��������ǽ��������ƶ��Ķ�ʱԤ�⡣ѡȡ��������ʱ�ε�GMS-5������ͼ������ͼ���򻮷�Ϊ32��32����
  //��ͼ���Ӽ������ý�����ط������ȡ������ͼ�����ƥ�����򣬸���ǰ����ͼƥ�������λ�ú�ʱ������ȷ
  //����ÿ��ͼ���Ӽ����ƶ�ʸ�����ٶȺͷ��򣩣�����ͼ���Ӽ����ƶ�ʸ�����п͹۷�������󣬻��ڼ�������
  //ͼ�ƶ�ʸ���������ú���켣��������ͼ����ʱ����Ԥ�⡣
  void normCrossCorrelation(const cv::Mat& img1, const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
  bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
public:
  LKTracker();
  //������ĸ��٣���
  bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,
                std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
  float getFB(){return fbmed;}
};

