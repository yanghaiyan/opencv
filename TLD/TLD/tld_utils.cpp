#include "tld_utils.h"
using namespace cv;
using namespace std;

/*vector��C++��׼ģ���STL�еĲ������ݣ�����һ���๦�ܵģ��ܹ������������ݽṹ���㷨��
ģ����ͺ����⡣vector֮���Ա���Ϊ��һ������������Ϊ���ܹ�������һ����Ÿ������͵Ķ���
�򵥵�˵��vector��һ���ܹ�����������͵Ķ�̬���飬�ܹ����Ӻ�ѹ�����ݡ�
Ϊ�˿���ʹ��vector�����������ͷ�ļ��а�������Ĵ��룺
#include <vector>
vector����std������ģ������Ҫͨ�������޶������������Ĵ��룺
using std::vector;
*/

void drawBox(Mat& image, CvRect box, Scalar color, int thick){
  rectangle( image, cvPoint(box.x, box.y), cvPoint(box.x+box.width,box.y+box.height),color, thick);
} 

//���� cvRound, cvFloor, cvCeil ��һ�����뷽�������븡����ת����������
//cvRound ���غͲ�����ӽ�������ֵ�� cvFloor ���ز����ڲ������������ֵ��
//cvCeil ���ز�С�ڲ�������С����ֵ��
void drawPoints(Mat& image, vector<Point2f> points,Scalar color){
  for( vector<Point2f>::const_iterator i = points.begin(), ie = points.end(); i != ie; ++i )
      {
      Point center( cvRound(i->x ), cvRound(i->y));  //������int i(3)�ĳ�ʼ������centerΪ��û�õ���
      circle(image,*i,2,color,1);
      }
}

Mat createMask(const Mat& image, CvRect box){
  Mat mask = Mat::zeros(image.rows,image.cols,CV_8U);
  drawBox(mask,box,Scalar::all(255),CV_FILLED);
  return mask;
}

//STL�е�nth_element()�����ҳ�һ��������������n���Ǹ�����
//��������a[0:len-1]����n������֣�����a[n],ͬʱa[0:n-1]��С��a[n],a[n+1:]������a[n],
//��a[n]���ҵ����������в�һ������
//������ֵ�������㷨�У�Ѱ����ֵ
float median(vector<float> v)
{
    int n = floor((double)v.size() / 2);
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

//<algorithm> //random_shuffle��ͷ�ļ�
//shuffle ϴ��  ���ȼ򵥵Ľ���һ���˿���ϴ�Ƶķ���������һ������ poker[52] �д���һ���˿�
//��1-52���Ƶ�ֵ��ʹ��һ��forѭ������������飬ÿ��ѭ��������һ��[0��52)֮��������RandNum��
//��RandNumΪ�����±꣬�ѵ�ǰ�±��Ӧ��ֵ��RandNum��Ӧλ�õ�ֵ������ѭ��������ÿ���ƶ���ĳ��
//λ�ý�����һ�Σ�����һ���ƾͱ������ˡ� ���������£�
/*
for (int i = 0; i < 52; ++i)  
{  
    int RandNum = rand() % 52;    
    int tmp = poker[i];  
    poker[i] = poker[RandNum];  
    poker[RandNum] = tmp;  
} 
*/
//��Ҫָ����Χ�ڵ����������ͳ�ķ�����ʹ��ANSI C�ĺ���random(),Ȼ���ʽ������Ա���������
//ָ���ķ�Χ�ڡ����ǣ�ʹ�������������������ȱ�㡣����ʽ��ʱ�����������Ť����,��ֻ֧����������
//C++���ṩ�˸��õĽ���������Ǿ���STL�е�random_shuffle()�㷨������ָ����Χ�ڵ����Ԫ�ؼ�����ѷ���
//�Ǵ���һ��˳�����У�Ҳ�������������������飩�������˳�������к���ָ����Χ������ֵ��
//���磬�������Ҫ����100��0-99֮���������ô�ʹ���һ����������100�����������е����������.
//���������֮����random_shuffle()�㷨����Ԫ������˳��
//Ĭ�ϵ�random_shuffle��, ���������е�index �� rand() % N ����λ�õ�ֵ���������ﵽ�����Ŀ�ġ�
//index_shuffle()���ڲ���ָ����Χ[begin:end]����������������������
vector<int> index_shuffle(int begin,int end){
  vector<int> indexes(end-begin);
  for (int i=begin;i<end;i++){
    indexes[i]=i;
  }
  random_shuffle(indexes.begin(),indexes.end());
  return indexes;
}

