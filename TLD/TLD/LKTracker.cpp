#include "LKTracker.h"
using namespace cv;

//������LK����������
//Media Flow ��ֵ�������� �� ���ٴ�����
//���캯������ʼ����Ա����
LKTracker::LKTracker(){
  ////���������Ҫ3��������һ�������ͣ��ڶ�������Ϊ�����������������һ�����ض�����ֵ��
  term_criteria = TermCriteria( TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03);
  window_size = Size(4,4);
  level = 5;
  lambda = 0.5;
}


bool LKTracker::trackf2f(const Mat& img1, const Mat& img2, vector<Point2f> &points1, vector<cv::Point2f> &points2){
  //TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
  //Forward-Backward tracking
  //����Forward-Backward Error����ֵ�����ٷ���
  //������LK����������
  //forward trajectory ǰ��켣����
  calcOpticalFlowPyrLK( img1,img2, points1, points2, status, similarity, window_size, level, term_criteria, lambda, 0);
  //backward trajectory ����켣����
  calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, FB_status,FB_error, window_size, level, term_criteria, lambda, 0);
  
  //Compute the real FB-error
  //ԭ��ܼ򵥣���tʱ�̵�ͼ���A�㣬���ٵ�t+1ʱ�̵�ͼ��B�㣻Ȼ�󵹻�������t+1ʱ�̵�ͼ���B�����ظ��٣�
  //������ٵ�tʱ�̵�ͼ���C�㣬�����Ͳ�����ǰ��ͺ��������켣���Ƚ�tʱ���� A�� �� C�� �ľ��룬�������
  //С��һ����ֵ����ô����Ϊǰ���������ȷ�ģ�����������FB_error
  //���� ǰ�� �� ���� �켣�����
  for( int i= 0; i<points1.size(); ++i ){
        FB_error[i] = norm(pointsFB[i]-points1[i]);   //norm()�����������ķ���??����ֵ��
  }
  //Filter out points with FB_error[i] <= median(FB_error) && points with sim_error[i] > median(sim_error)
  normCrossCorrelation(img1, img2, points1, points2);
  return filterPts(points1, points2);
}

//����NCC�Ѹ���Ԥ��Ľ����Χȡ10*10��СͼƬ��ԭʼλ����Χ10*10��СͼƬ��ʹ�ú���getRectSubPix�õ�����
//��ģ��ƥ�䣨����matchTemplate��
void LKTracker::normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
        Mat rec0(10,10,CV_8U);
        Mat rec1(10,10,CV_8U);
        Mat res(1,1,CV_32F);

        for (int i = 0; i < points1.size(); i++) {
                if (status[i] == 1) {  //Ϊ1��ʾ����������ٳɹ�
						//��ǰһ֡�͵�ǰ֡ͼ���У���ÿ��������Ϊ���ģ�����ȡ10x10���ؾ��Σ�ʹ�������ؾ���
                        getRectSubPix( img1, Size(10,10), points1[i],rec0 );   
                        getRectSubPix( img2, Size(10,10), points2[i],rec1);
						//ƥ��ǰһ֡�͵�ǰ֡����ȡ��10x10���ؾ��Σ��õ�ƥ����ӳ��ͼ��
						//CV_TM_CCOEFF_NORMED ��һ�����ϵ��ƥ�䷨
						//�����ֱ�Ϊ����������ͼ������ģ�塣�ȽϽ����ӳ��ͼ��ָ��ƥ�䷽��
                        matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED); 
                        similarity[i] = ((float *)(res.data))[0];  //�õ���������������ƶȴ�С

                } else {
                        similarity[i] = 0.0;
                }
        }
        rec0.release();
        rec1.release();
        res.release();
}

//ɸѡ�� FB_error[i] <= median(FB_error) �� sim_error[i] > median(sim_error) ��������
//�õ�NCC��FB error�������ֵ���ֱ�ȥ����ֵһ��ĸ��ٽ�����õĵ�
bool LKTracker::filterPts(vector<Point2f>& points1,vector<Point2f>& points2){
  //Get Error Medians
  simmed = median(similarity);   //�ҵ����ƶȵ���ֵ
  size_t i, k;
  for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
          continue;
        if(similarity[i]> simmed){   //ʣ�� similarity[i]> simmed ��������
          points1[k] = points1[i];   
          points2[k] = points2[i];
          FB_error[k] = FB_error[i];
          k++;
        }
    }
  if (k==0)
    return false;
  points1.resize(k);
  points2.resize(k);
  FB_error.resize(k);

  fbmed = median(FB_error);     //�ҵ�FB_error����ֵ
  for( i=k = 0; i<points2.size(); ++i ){
      if( !status[i])
        continue;
      if(FB_error[i] <= fbmed){   
        points1[k] = points1[i];   //�ٶ���һ��ʣ�µ��������һ��ɸѡ��ʣ�� FB_error[i] <= fbmed ��������
        points2[k] = points2[i];
        k++;
      }
  }
  points1.resize(k);
  points2.resize(k);
  if (k>0)
    return true;
  else
    return false;
}




/*
 * old OpenCV style
void LKTracker::init(Mat img0, vector<Point2f> &points){
  //Preallocate
  //pyr1 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //pyr2 = cvCreateImage(Size(img1.width+8,img1.height/3),IPL_DEPTH_32F,1);
  //const int NUM_PTS = points.size();
  //status = new char[NUM_PTS];
  //track_error = new float[NUM_PTS];
  //FB_error = new float[NUM_PTS];
}


void LKTracker::trackf2f(..){
  cvCalcOpticalFlowPyrLK( &img1, &img2, pyr1, pyr1, points1, points2, points1.size(), window_size, level, status, track_error, term_criteria, CV_LKFLOW_INITIAL_GUESSES);
  cvCalcOpticalFlowPyrLK( &img2, &img1, pyr2, pyr1, points2, pointsFB, points2.size(),window_size, level, 0, 0, term_criteria, CV_LKFLOW_INITIAL_GUESSES | CV_LKFLOW_PYR_A_READY | CV_LKFLOW_PYR_B_READY );
}
*/

