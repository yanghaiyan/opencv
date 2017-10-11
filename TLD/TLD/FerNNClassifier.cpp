
#include "FerNNClassifier.h"

using namespace cv;
using namespace std;

void FerNNClassifier::read(const FileNode& file){
  ///Classifier Parameters
  //������Щ����ͨ������ʼ����ʱ����parameters.yml�ļ����г�ʼ��
  valid = (float)file["valid"];
  ncc_thesame = (float)file["ncc_thesame"];
  nstructs = (int)file["num_trees"];   //��ľ����һ�������鹹����ÿ����������ͼ���Ĳ�ͬ��ͼ��ʾ���ĸ���
  structSize = (int)file["num_features"];  //ÿ����������������Ҳ��ÿ�����Ľڵ����������ÿһ����������Ϊһ�����߽ڵ�
  thr_fern = (float)file["thr_fern"];
  thr_nn = (float)file["thr_nn"];
  thr_nn_valid = (float)file["thr_nn_valid"];
}

void FerNNClassifier::prepare(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs * structSize;
  //��ά����  ����ȫ���߶ȣ�scales����ɨ�贰�ڣ�ÿ���߶Ȱ���totalFeatures������
  features = vector<vector<Feature> >(scales.size(), vector<Feature> (totalFeatures));
 
  //opencv���Դ���һ�����������������RNG
  RNG& rng = theRNG();
  
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  //���Ϸ���������n��������������ÿ�����������ǻ���һ��pixel comparisons�����رȽϼ����ģ�
  //pixel comparisons�Ĳ�������������һ����һ����patchȥ��ɢ�����ؿռ䣬�������п��ܵĴ�ֱ��ˮƽ��pixel comparisons
  //Ȼ�����ǰ���Щpixel comparisons��������n����������ÿ���������õ���ȫ��ͬ��pixel comparisons���������ϣ���
  //���������з�������������ͳһ�����Ϳ��Ը�������patch��
  
  //�������ȥ���ÿһ���߶�ɨ�贰�ڵ�����
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng;
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0; s<scales.size(); s++){
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
		  //��s�ֳ߶ȵĵ�i������  ���������������ص�����
          features[s][i] = Feature(x1, y1, x2, y2);
      }
  }
  //Thresholds
  thrN = 0.5 * nstructs;

  //Initialize Posteriors  ��ʼ���������
  //�������ָÿһ���������Դ����ͼ��Ƭ�������ضԱȣ�ÿһ�����ضԱȵõ�0����1�����е�����13��comparison�Աȣ�
  //����һ��13λ�Ķ����ƴ���x��Ȼ��������һ����¼�˺�����ʵ�����P(y|x)��yΪ0����1�������ࣩ��Ҳ���ǳ���x��
  //�����ϣ���ͼ��ƬΪy�ĸ����Ƕ��ٶ�n�������������ĺ��������ƽ��������0.5���ж��京��Ŀ��
  for (int i = 0; i<nstructs; i++) {
  //ÿһ��ÿ����ά��һ��������ʵķֲ�������ֲ���2^d����Ŀ��entries��������d�����رȽ�pixel comparisons
  //�ĸ�����������structSize����13��comparison�����Ի����2^13��8,192�����ܵ�code��ÿһ��code��Ӧһ���������
  //�������P(y|x)= #p/(#p+#n) ,#p��#n�ֱ������͸�ͼ��Ƭ����Ŀ��Ҳ���������pCounter��nCounter
  //��ʼ��ʱ��ÿ��������ʶ��ó�ʼ��Ϊ0������ʱ�������淽ʽ���£���֪����ǩ��������ѵ��������ͨ��n��������
  //���з��࣬���������������ô��Ӧ��#p��#n�ͻ���£�����P(y|x)Ҳ��Ӧ������
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}

//�ú����õ������image���������Ľڵ㣬Ҳ�����������������13λ�Ķ����ƴ��룩
void FerNNClassifier::getFeatures(const cv::Mat& image, const int& scale_idx, vector<int>& fern){
  int leaf;  //Ҷ��  �������սڵ�
  //ÿһ��ÿ����ά��һ��������ʵķֲ�������ֲ���2^d����Ŀ��entries��������d�����رȽ�pixel comparisons
  //�ĸ�����������structSize����13��comparison�����Ի����2^13��8,192�����ܵ�code��ÿһ��code��Ӧһ���������
  for (int t=0; t<nstructs; t++){  //nstructs ��ʾ���ĸ��� 10
      leaf=0;
      for (int f=0; f<structSize; f++){  //��ʾÿ���������ĸ��� 13
	    //struct Feature �����ṹ����һ����������� bool operator ()(const cv::Mat& patch) const
		//���ص�patchͼ��Ƭ��(y1,x1)��(y2, x2)������رȽ�ֵ������0����1
		//Ȼ��leaf�ͼ�¼����13λ�Ķ����ƴ��룬��Ϊ����
          leaf = (leaf << 1) + features[scale_idx][t*nstructs+f](image);
      }
      fern[t] = leaf; 
  }
}

float FerNNClassifier::measure_forest(vector<int> fern) {
  float votes = 0;
  for (int i = 0; i < nstructs; i++) {
     // �������posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      votes += posteriors[i][fern[i]];   //ÿ������ÿ������ֵ��Ӧ�ĺ�������ۼ�ֵ ��ͶƱֵ����
  }
  return votes;
}

//����������������ͬʱ���º������
void FerNNClassifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int i = 0; i < nstructs; i++) {
      idx = fern[i];
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) {
          posteriors[i][idx] = 0;
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
}

//ѵ�����Ϸ�������n���������������ϣ�
void FerNNClassifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
  
  //thr_fern: 0.6 thrP����ΪPositive thershold
  thrP = thr_fern * nstructs;                                    // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index
                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].second==1){    //Ϊ1��ʾ������        //       if (Y[I] == 1) {
		      //measure_forest������������������������ֵ��Ӧ�ĺ�������ۼ�ֵ
			  //���ۼ�ֵ���С����������ֵ��Ҳ���������������������ȴ������ɸ�������
			  //���ַ���������ԾͰѸ�������ӵ��������⣬ͬʱ���º������
              if(measure_forest(ferns[i].first) <= thrP)      //         if (measure_forest(x) <= thrP)
			  ////��������������ͬʱ���º������
                update(ferns[i].first, 1, 1);                 //             update(x,1,1);
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first, 0, 1);                 //             update(x,0,1);
          }
      }
  //}
}

//ѵ������ڷ�����
void FerNNClassifier::trainNN(const vector<cv::Mat>& nn_examples){
  float conf, dummy;
  vector<int> y(nn_examples.size(),0); //vector<T> v3(n, i); v3����n��ֵΪi��Ԫ�ء�y����Ԫ�س�ʼ��Ϊ0
  y[0]=1;  //����˵������trainNN������������nn_data��������ֻ��һ��pEx����nn_data[0]
  vector<int> isin;
  for (int i=0; i<nn_examples.size(); i++){                          //  For each example
      //��������ͼ��Ƭ������ģ��֮���������ƶ�conf
      NNConf(nn_examples[i], isin, conf, dummy);                      //  Measure Relative similarity
	  //thr_nn: 0.65 ��ֵ
	  //��ǩ�������������������ƶ�С��0.65 ������Ϊ�䲻����ǰ��Ŀ�꣬Ҳ���Ƿ�������ˣ���ʱ��Ͱ����ӵ���������
      if (y[i]==1 && conf <= thr_nn){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65
          if (isin[1]<0){                                          //      if isnan(isin(2))
              pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
              continue;                                            //        continue;
          }                                                        //      end
          //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
          pEx.push_back(nn_examples[i]);
      }                                                            //    end
      if(y[i]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
        nEx.push_back(nn_examples[i]);                             //    tld.nex = [tld.nex x(:,i)];

  }                                                                 //  end
  acum++;
  printf("%d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                  //  end

  /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf)������ƶ�, Conservative Similarity (csconf)�������ƶ�,
   * In pos. set|Id pos set|In neg. set (isin)
   */
void FerNNClassifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  isin=vector<int>(3,-1);  //vector<T> v3(n, i); v3����n��ֵΪi��Ԫ�ء� ����Ԫ�ض���-1
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0; //    conf1 = zeros(1,size(x,2));
      csconf=0;
      return;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP, csmaxP, maxP=0;
  bool anyP=false;
  int maxPidx, validatedPart = ceil(pEx.size()*valid);  //ceil���ش��ڻ��ߵ���ָ�����ʽ����С����
  float nccN, maxN=0;
  bool anyN=false;
  //�Ƚ�ͼ��Ƭp������ģ��M�ľ��루���ƶȣ���������������������ƶȣ�Ҳ���ǽ������ͼ��Ƭ��
  //����ģ�������е�ͼ��Ƭ����ƥ�䣬�ҳ������Ƶ��Ǹ�ͼ��Ƭ��Ҳ�������ƶȵ����ֵ
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i], example, ncc, CV_TM_CCORR_NORMED);      // measure NCC to positive examples
      nccP=(((float*)ncc.data)[0]+1)*0.5;  //����ƥ�����ƶ�
      if (nccP>ncc_thesame)  //ncc_thesame: 0.95
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;    //��¼�������ƶ��Լ���Ӧ��ͼ��Ƭindex����ֵ
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
  }
  //���㸺������������ƶ�
  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  //set isin
  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  if (anyP) isin[0]=1;  
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  if (anyN) isin[2]=1; 
  
  //Measure Relative Similarity
  //������ƶ� = ��������������ƶ� / ����������������ƶ� + ��������������ƶȣ�
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
}

void FerNNClassifier::evaluateTh(const vector<pair<vector<int>,int> >& nXT, const vector<cv::Mat>& nExT){
  float fconf;
  for (int i=0;i<nXT.size();i++){
  //���л����������ĺ�����ʵ�ƽ��ֵ�������thr_fern������Ϊ����ǰ��Ŀ��
  //measure_forest���ص������к�����ʵ��ۼӺͣ�nstructs Ϊ���ĸ�����Ҳ���ǻ�������������Ŀ ����
    fconf = (float) measure_forest(nXT[i].first)/nstructs;
    if (fconf>thr_fern)  //thr_fern: 0.6 thrP����ΪPositive thershold
      thr_fern = fconf;  //ȡ���ƽ��ֵ��Ϊ �ü��Ϸ������� �µ���ֵ�������ѵ������
  }
  
  vector <int> isin;
  float conf, dummy;
  for (int i=0; i<nExT.size(); i++){
      NNConf(nExT[i], isin, conf, dummy);
      if (conf > thr_nn)
        thr_nn = conf; //ȡ������������ƶ���Ϊ ������ڷ������� �µ���ֵ�������ѵ������
  }
  
  if (thr_nn > thr_nn_valid)  //thr_nn_valid: 0.7
    thr_nn_valid = thr_nn;
}

//���������⣨����ģ�ͣ�������������������ʾ�ڴ�����
void FerNNClassifier::show(){
  Mat examples((int)pEx.size()*pEx[0].rows, pEx[0].cols, CV_8U);
  double minval;
  Mat ex(pEx[0].rows, pEx[0].cols, pEx[0].type());
  for (int i=0;i<pEx.size();i++){
    //minMaxLocѰ�Ҿ���һά���鵱����������Mat���壩����Сֵ�����ֵ��λ��. 
    minMaxLoc(pEx[i], &minval); //Ѱ��pEx[i]����Сֵ
    pEx[i].copyTo(ex);
    ex = ex - minval;  //������������С����������Ϊ0���������ذ�������
	//Mat Mat::rowRange(int startrow, int endrow) const Ϊָ������span����һ���µľ���ͷ��
	//Mat Mat::rowRange(const Range& r) const   //Range �ṹ��������ʼ����ֹ������ֵ��
    Mat tmp = examples.rowRange(Range(i*pEx[i].rows, (i+1)*pEx[i].rows));
    ex.convertTo(tmp, CV_8U);
  }
  imshow("Examples", examples);
}
