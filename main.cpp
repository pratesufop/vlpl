#include <iostream>
#include "cvaux.h"
#include "cv.h"
#include <ios>
#include "highgui.h"
#include <vector>
#include <fstream>
#include <ctype.h>
#include <dirent.h>
#include <stdexcept>
#include <ctype.h>
#include "Thog.h"
#include "HOGNMS.h"
#include "HOG_Result.h"
#include <pthread.h>
#include <unistd.h>
using namespace std;
using namespace cv;


struct ROI
{
    float pertinencia;
    cv::Rect roi;
    int label;
};

struct Dados
{
    vector<double> n_values;
    cv::Mat img;
    vector<HOG_Result> meus_roi;
    int scale;
    string endereco;
    int currentFile;
    double *scales;
    int stride;
    int nwindows;
};

struct Dados_Limiar
{
    string nome_file;
    int currentFile;
    vector<HOG_Result>n_roi;
    int limiar;
};

void getFilesInDirectory(vector<string>& dirName, vector<string>&fileNames, const vector<string>& validExtensions, int *nfiles_dir);
void setNegative(vector<string>& e_neg, bool train);
void setPositive(vector<string>& e_pos, bool train);
void showDetections(const vector<Rect>& found, Mat& imageData, int n);
void Avaliar( vector<HOG_Result> *meus_ROI, string image_file, int indice_imagem, int cont_windows, int *positive_window);
double Area(Rect r);
void Carregar_Dados(vector<string> dir);
void TrainHOG();
void Train_HOG(bool cross_validation, int n_folds, char *savexml);
void Inicialiar_SVM();
void calculateFeaturesFromInput(const string& imageFilename, vector<float> &featureVector);
int getNumberofFeatures();
void calculateMeusFeatures(const string& imageFilename, vector<float> &featureVector);
void Teste_MeuHOG(char *n_teste);
void get_hogdescriptor_visu(cv::Mat img,double *descriptorValues, double zoomFac, double angle, int nbins);
bool sortbyPertinencia(const HOG_Result &a, const HOG_Result &b);
void IHog_MultiDetect(char *model_file, int n_scale, double *scale);
void Supression(int ind, vector<HOG_Result> *meus_roi, int *detected);
void* Processa_Escala(void * arg);
void * Processa_Limiar(void * arg);
void TestePedro();

fstream janela_out;

char *end_Minha="Hog_minha.xml";
cv::Mat img_crop, ROI_image;
cv::Rect cropped_rect;
int select_flag=0, drag=0;
vector<string> nome_dir_teste;
Point point1, point2;
void Teste_GPU();

char nome_dir_test_pos[300]="C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-UFOP-gate/102TESTE/rot_vlp/";
char nome_dir_test_neg[300]="C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-UFOP-gate/102TESTE/rot_vlp/";
char nome_dir_test[300]="C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-UFOP-gate/101TESTE/";
char nome_dir_out[300]="imgs-LPIV/imgs-portaria-UFOP/RESULTADO_TREINAMENTO/Teste";
char nome_dir_sub[300]="imgs-LPIV/imgs-portaria-UFOP/SubImages/Teste";
char sub_dir[300]="imgs-LPIV/imgs-portaria-UFOP/SubImages/";
int *targets;
int *targets_global;
int *labels;
int *train_images;
int n_images;
double **meus_valores;
double *valores_globais;
int n_imagens_validacao;
vector<string> positive_example;//endereço das imagens de treino positivas
vector<string> negative_example;//end. das img de treino negativas.
vector<string> validacao_positivos;//endereço do conjunto de validação (positivas)
vector<string> validacao_negativos;//endereço do conjunto de validação (negativas)
double prob_placa;
double prob_nonplaca;
int n_combinacoes;
double threshold_inf_L1=1, threshold_sup_L1=0, threshold_inf_L2=1, threshold_sup_L2=0;
int window_l=36;
int window_c=108;
vector<double> x,y,w,h;
CvSVM svm;
cv::SVMParams params;
Size *Block_size;
Size *Cell_size;
Size *Window_size;
int nbins;
cv::Mat img1;
fstream fout;
int total_janelas=0;
vector<double> nvalues_mult;
cv::Mat img_multi;
vector<HOG_Result> meus_roi_multi;

int n_scale_multi=11;
double scale_multi[11]={0.7,0.75,0.8,0.85,0.9,1.0,1.1,1.15,1.2,1.25,1.3};
int n_limiares=5;
double limiares[5]={0.5,0.6,0.7, 0.80, 0.90};

pthread_mutex_t mutex_for_some_value = PTHREAD_MUTEX_INITIALIZER;
double **tp, **fp;
int n_thresholds=1;
const int number_of_threads=n_scale_multi;
///para placa....
int tamanho_cellx=4;
int tamanho_celly=4;

int tamanho_blocox=2;
int tamanho_blocoy=2;

double *meu_thresh;
int *files_by_dir1,*files_by_dir2;
int dx, dy; //parâmetro que indica a qtde de fundo a ser adicionado...
vector<string> dir_limiares;
int global_stride=9;

int main()
{
    dx=9, dy=3;
    cv::Mat image, nimage,image_raw;
    vector<float> f;
    cv::Rect **minhas_regioes;
    int n_regioes=2;
    int param1=1;
    cv::Mat n=imread("DCAM0065.png",0);

    n_images=0;
    n_imagens_validacao=0;

    Block_size= new Size(tamanho_blocox,tamanho_blocoy);
    Cell_size=new Size(tamanho_cellx,tamanho_celly);
    Window_size=new Size(window_c,window_l);
    janela_out.open("Janelas.txt",ios::out);
    vector<string> nome_dir_pos;
    vector<string> nome_dir_neg;
    vector<string> validation_dir_pos;
    vector<string> validation_dir_neg;
    vector<string> exemplos_positivos;
    vector<string> exemplos_negativos;
    static vector<string> ext_valida;
    ext_valida.push_back("png");
    ext_valida.push_back("jpg");

    nome_dir_pos.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-UFOP-gate/101PHOTO/rot_vlp/positivas/");
    nome_dir_neg.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-UFOP-gate/101PHOTO/rot_vlp/negativas/");

    nbins=9;
    int angulo=180;
    int nlevels=1;
    Thog hogt(nbins,1.0,n.rows,n.cols,0.02,angulo,nlevels,false,true,true);
    hogt.setWindow_size(Size(window_c,window_l));
    hogt.setBlock(Size(tamanho_blocox,tamanho_blocoy));
    hogt.setCellSize(Size(tamanho_cellx,tamanho_celly));

    int option;
    cout<<"Validacao usando minha implementacao [10]"<<endl;
    cout<<"Teste usando minha implementacao [16]"<<endl;
    cout<<"Treinar usando minha implementacao HOG [20]"<<endl;
    cout<<"Integral HOG MultiScale Teste         [21]"<<endl;
    cout<<"Testar metodo do Pedro... [24]"<<endl;

    cin>>option;
    else if(option==10)
    {
        vector<string> pos_base_dir;
        strcat(nome_dir_test_pos,"treino_positivas/");
        pos_base_dir.push_back(nome_dir_test_pos);
        files_by_dir1=new int[pos_base_dir.size()];
        getFilesInDirectory(pos_base_dir,exemplos_positivos, ext_valida,files_by_dir1);

        vector<string> neg_base_dir;
        strcat(nome_dir_test_neg,"treino_negativas/");
        neg_base_dir.push_back(nome_dir_test_neg);
        files_by_dir2=new int[neg_base_dir.size()];
        getFilesInDirectory(neg_base_dir,exemplos_negativos, ext_valida,files_by_dir2);
        setPositive(exemplos_positivos,true);
        setNegative(exemplos_negativos,true);

        Teste_MeuHOG(end_Minha);

    }
    else if(option==16)
    {
        vector<string> pos_base_dir;
        strcat(nome_dir_test_pos,"positivas/");
        pos_base_dir.push_back(nome_dir_test_pos);
        files_by_dir1=new int[pos_base_dir.size()];
        getFilesInDirectory(pos_base_dir,exemplos_positivos, ext_valida,files_by_dir1);

        vector<string> neg_base_dir;
        strcat(nome_dir_test_neg,"negativas/");
        neg_base_dir.push_back(nome_dir_test_neg);
        files_by_dir2=new int[neg_base_dir.size()];
        getFilesInDirectory(neg_base_dir,exemplos_negativos, ext_valida,files_by_dir2);

        setPositive(exemplos_positivos,true);
        setNegative(exemplos_negativos,true);

        Teste_MeuHOG(end_Minha);
    }
    else if(option==20)
    {
        files_by_dir1=new int[nome_dir_pos.size()];
        files_by_dir2=new int[nome_dir_neg.size()];
        getFilesInDirectory(nome_dir_pos,exemplos_positivos, ext_valida,files_by_dir1);
        getFilesInDirectory(nome_dir_neg,exemplos_negativos, ext_valida,files_by_dir2);
        setPositive(exemplos_positivos,true);
        setNegative(exemplos_negativos,true);
        Train_HOG(true,10, end_Minha);

    }
     else if(option==21)
    {

        vector<string> pos_base_dir;
        pos_base_dir.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-UFOP-gate/101TESTE/");
        files_by_dir1=new int[pos_base_dir.size()];
        double scale[10]={0.80, 0.90,0.95,1.05,1.10,1.15, 1.20, 1.27, 1.33, 1.40 };
        getFilesInDirectory(pos_base_dir,exemplos_positivos, ext_valida,files_by_dir1);
        setPositive(exemplos_positivos,true);
        IHog_MultiDetect(end_Minha,10,scale);

    }
    else if(option==24)
    {
        vector<string> pos_base_dir;
        pos_base_dir.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-UFOP-gate/101TESTE/");
        files_by_dir1=new int[pos_base_dir.size()];
        getFilesInDirectory(pos_base_dir,exemplos_positivos, ext_valida,files_by_dir1);

        setPositive(exemplos_positivos,true);
        TestePedro();


    }
    cout<<"Fim !"<<endl;

}


void getFilesInDirectory(vector<string>& dirName, vector<string>&fileNames, const vector<string>& validExtensions, int *nfiles_dir)
{
    printf("Opening directory %s\n", dirName.back().c_str());
    struct dirent* ep;
    size_t extensionLocation;
    for(int i=0;i<dirName.size();i++)
    {
        DIR* dp = opendir((dirName.at(i)).c_str());
        cout<<dirName.at(i)<<endl;
        //getchar();
        nfiles_dir[i]=0;
        if (dp != NULL)
        {
            while ((ep = readdir(dp)))
            {
                if(ep==NULL)
                {
                    break;
                }
                extensionLocation = string(ep->d_name).find_last_of(".");
                string tempExt =(string(ep->d_name).substr(extensionLocation + 1));

                for(int k=0;k<tempExt.size();k++)
                     tempExt[k]=tolower(tempExt[k]);

                if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end())
                {
                    fileNames.push_back((string) dirName.at(i) + ep->d_name);
                    nfiles_dir[i]=nfiles_dir[i]+1;
                }
            }
            (void) closedir(dp);
        }
        else
        {
            printf("Error opening directory '%s'!\n", (dirName.at(i)).c_str());
        }
    }
    return;
}



void showDetections(const vector<Rect>& found, Mat& imageData, int n)
{
    vector<Rect> found_filtered;
    size_t i, j;
    for (i = 0; i < found.size(); ++i)
    {
        Rect r = found[i];
        for (j = 0; j < found.size(); ++j)
            if (j != i && (r & found[j]) == r)
                break;
        if (j == found.size())
            found_filtered.push_back(r);
    }
    for (i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        if(n==0)//blue
            rectangle(imageData, r.tl(), r.br(), Scalar(255,0, 0), 4);
        else if(n==1)//yellow
            rectangle(imageData, r.tl(), r.br(), Scalar(0,255, 255), 4);
        else if(n==2)//Red
            rectangle(imageData, r.tl(), r.br(), Scalar(0,0, 255), 4);
        else if(n==3)//hot pink
            rectangle(imageData, r.tl(), r.br(), Scalar(180,255,105), 4);
    }
}

void Avaliar( vector<HOG_Result> *meus_ROI, string image_file, int indice_imagem, int cont_windows, int *positive_window)
{

    int n_positivos=0;
    cout<<indice_imagem<<endl;
    cout<<x.at(indice_imagem)<<" , "<<y.at(indice_imagem)<<" , "<<w.at(indice_imagem)<<" , "<<h.at(indice_imagem)<<endl;

    cv::Rect *referencia=new cv::Rect(x.at(indice_imagem),y.at(indice_imagem),w.at(indice_imagem),h.at(indice_imagem));

    for(int i=0;i<cont_windows;i++)
    {
        Rect *f=new Rect((meus_ROI->at(i)).roi);
        double *uniao=new double(), *intersecao=new double();
        *intersecao=Area(*referencia&*f);
        *uniao= Area(*referencia)+Area(*f)- *intersecao;
        for(int j=0;j<n_limiares;j++)
        {
                 if(*intersecao/ *uniao > limiares[j])
                {
                     (meus_ROI->at(i)).label[j]=1;
                     *positive_window=*positive_window+1;
                }
                else
                    ((meus_ROI)->at(i)).label[j]=-1;
        }
        delete f, intersecao, uniao;
    }
    delete referencia;

}
void Carregar_Dados(vector<string> dir)
{

    string aux;
    int n_elementos= dir.size();
    for(int i=0;i<n_elementos;i++)
    {

        aux=dir.at(i);
        fstream fin;
        fin.open(aux.c_str(),ios::in);
        if(fin.is_open())
            cout<<"arquivo de avaliacao aberto com sucesso!"<<endl;

        size_t extensionLocation;

        int n=1;
        string aux;
        while(getline(fin,aux))
        {
            extensionLocation = aux.find_last_of(",");
            string tempExt =(aux.substr(extensionLocation + 1));
            h.push_back(atof(tempExt.c_str()));

            string tempExt2=(aux.substr(0,extensionLocation));
            extensionLocation=tempExt2.find_last_of(",");
            tempExt =(aux.substr(extensionLocation + 1));
            w.push_back(atof(tempExt.c_str()));

            tempExt2=(aux.substr(0,extensionLocation));
            extensionLocation=tempExt2.find_last_of(",");
            tempExt =(aux.substr(extensionLocation + 1));
            tempExt2=(aux.substr(0,extensionLocation));

            y.push_back(atof(tempExt.c_str()));
            x.push_back(atof(tempExt2.c_str()));
         }
    }
}

void Train_HOG(bool cross_validation, int n_folds, char *savexml)
{
    Inicialiar_SVM();
    int total_features;
    printf("Beginning to extract HoG features from images\n");
    int t_samples=n_images;
    float *targets=new float[t_samples];
    cout<<"Total samples: "<<t_samples<<endl;
    cout<<"Positive examples: "<<positive_example.size()<<endl;
    cout<<"Negative examples: "<<negative_example.size()<<endl;
    getchar();
    double **hog_values=new double*[t_samples];

    for (unsigned long currentFile = 0; currentFile < t_samples; ++currentFile)
    {
        cout<<currentFile<<endl;
        Thog *meu_hog=new Thog(9,1.0,window_l,window_c,0.02,180,1,false,true,true);

        meu_hog->setWindow_size(Size(window_c,window_l));
        meu_hog->setBlock(Size(tamanho_blocox,tamanho_blocoy));
        meu_hog->setCellSize(Size(tamanho_cellx,tamanho_celly));
        Size window=meu_hog->getWindowSize();

        total_features=meu_hog->getNumberFeatures();
        const string currentImageFile =  (currentFile < positive_example.size() ? positive_example.at(currentFile) : negative_example.at(currentFile - positive_example.size()));
        cout<<currentImageFile<<endl;
        if(currentFile < positive_example.size())
        {
            targets[currentFile]=1;
        }
        else
        {
            targets[currentFile]=2;
        }
        cv::Mat img = imread(currentImageFile,0);
        cv::resize(img,img,*Window_size);

        meu_hog->CalculateIntegralHOG(img,false);

        hog_values[currentFile]=new double[total_features];

        meu_hog->calculateHOG_window(Rect(0, 0,window.width, window.height),hog_values[currentFile]);

        img.release();
        delete meu_hog;
     }


     cv::Mat *trainData=new cv::Mat(t_samples, total_features, CV_32F);


    for (unsigned long currentFile = 0; currentFile < t_samples; ++currentFile)
    {
            for (int j = 0; j < total_features; j++)
            {
                    trainData->at<float>(currentFile,j)=float(hog_values[currentFile][j]);
        }
    }

     cout<<"N features: "<<trainData->cols<<endl;
     cout<<"N samples: "<<trainData->rows<<endl;
     cv::Mat labels(t_samples,1,CV_32F,targets);

     cout<<"Realizando treinamento..."<<endl;
     if(!cross_validation)
            svm.train(*trainData, labels, Mat(), Mat(), params);
    else
            svm.train_auto(*trainData, labels, Mat(), Mat(), params, n_folds);

    cout<<"Fim treinamento... Salvando classificador SVM"<<endl;
    svm.save(savexml);
    cout<<"Deletando elementos..."<<endl;
    delete []targets;
    delete []hog_values;
    trainData->release();
    labels.release();
}

void Inicialiar_SVM()
{
    params.svm_type    = CvSVM::C_SVC;
    params.C=1;
    params.kernel_type = CvSVM::LINEAR;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER||CV_TERMCRIT_EPS , 1000, 1e-6);
    CvMat* weights = cvCreateMat( 1, 2, CV_32FC1 );
    CV_MAT_ELEM( *weights, float, 0, 0 ) = 10.0;
    CV_MAT_ELEM( *weights, float, 0, 1 ) = 0.1;
    params.class_weights=weights;
}

void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector)
{
    int h_size, v_size;
    h_size=Block_size->width*Cell_size->width;
    v_size=Block_size->height*Cell_size->height;
    HOGDescriptor *hog;
    hog=new HOGDescriptor(*Window_size,Size(h_size,v_size), Size(tamanho_cellx,tamanho_celly),Size(tamanho_blocox,tamanho_blocoy),nbins,1,0.2,0,64);

    Mat entrada = imread(imageFilename,0);
    Size trainingPadding = Size(0, 0);
    int nh,nv;
    nh=Block_size->width*Cell_size->width/2;
    nv=Block_size->height*Cell_size->height/2;
    Size winStride = Size(nh, nv);

    if (entrada.empty())
    {
        featureVector.clear();
        printf("Error: HOG image '%s' is empty, features calculation skipped!\n", imageFilename.c_str());
        return;
    }

    if (entrada.cols != hog->winSize.width || entrada.rows != hog->winSize.height)
    {
        featureVector.clear();
        printf("Error: Image '%s' dimensions (%u x %u) do not match HOG window size (%u x %u)!\n", imageFilename.c_str(), entrada.cols, entrada.rows, hog->winSize.width, hog->winSize.height);
        return;
    }
    vector<Point> locations;
    hog->compute(entrada, featureVector, winStride, trainingPadding, locations);
    entrada.release();
    delete hog;
}

int getNumberofFeatures()
{
    int n_features;
    int nb_h=((Window_size->width-Cell_size->width*Block_size->width)/Cell_size->width)+1;
    int nb_v=((Window_size->height-Cell_size->height*Block_size->height)/Cell_size->height)+1;
    n_features=nb_h*nb_v*nbins*(Block_size->height*Block_size->width);

    return n_features;
}
void calculateMeusFeatures(const string& imageFilename, vector<float> &featureVector)
{
    Mat entrada = imread(imageFilename);
    cv::Mat *my_image=new cv::Mat(window_l,window_c,CV_64F);
    cv::resize(entrada,*my_image,my_image->size());
    Thog *meu_hog=new Thog(9,1.0,window_l,window_c,0.02,180,1,false,true,true);

    meu_hog->setWindow_size(Size(window_c,window_l));
    meu_hog->setBlock(Size(tamanho_blocox,tamanho_blocoy));
    meu_hog->setCellSize(Size(tamanho_cellx,tamanho_celly));
    CvSize window=meu_hog->getWindowSize();
    int total_features=meu_hog->getNumberFeatures();
    double *atributos = new double[total_features];
    meu_hog->CalculateIntegralHOG(*my_image,false);

    Rect sub_image(0,0,window.width,window.height);
    cv::Mat subimage(*my_image,sub_image);
    subimage.convertTo(subimage,CV_8U);
    //calculando o descritor HOG com base na minha implementação.
    meu_hog->calculateHOG_window(sub_image,atributos);
    subimage.release();

    for(int i=0; i<total_features;i++)
        featureVector.push_back(float(atributos[i]));
    my_image->release();

    entrada.release(); // Release the image again after features are extracted
    delete []atributos;
    delete meu_hog;
}

void Teste_MeuHOG(char *n_teste)
{
    svm.load(n_teste);
    int h_size, v_size;
    h_size=Block_size->width*Cell_size->width;
    v_size=Block_size->height*Cell_size->height;

    int t_samples=n_images;

    float tp=0;//true positive
    float fn=0;//false negative.
    float tn=0;//true negative
    float fp=0;
    vector<double> nvalues;

    for (unsigned long currentFile = 0; currentFile < t_samples; ++currentFile)
    {
        const string currentImageFile=(currentFile < positive_example.size() ? positive_example.at(currentFile) : negative_example.at(currentFile - positive_example.size()));
        vector<float> w_hog;
        calculateMeusFeatures(currentImageFile,w_hog);
        cv::Mat *test_data=new cv::Mat(cv::Mat::zeros(1, getNumberofFeatures(), CV_32F));
        for(int i=0;i<w_hog.size();i++)
        {
             test_data->at<float>(0,i)=w_hog.at(i);
        }

        nvalues.push_back(svm.predict(*test_data,true));
        if(currentFile< positive_example.size())//classe real positiva
        {
            if(nvalues.back()>0)
            {
                tp++;
            }
            else
            {
                fn++;
            }
        }
        else
        {
            if(nvalues.back()<0)
                tn++;
            else
                fp++;
        }
        w_hog.clear();
        test_data->release();
    }
    cout<<"                     Confusion Matrix             "<<endl;
    cout<<"                      Classe Real                 "<<endl;
    cout<<"                 Positive    |     Negative        "<<endl;
    cout<<"Positive:       "<<tp<<"          |       "<<fp<<endl;
    cout<<"Negative:       "<<fn<<"          |       "<<tn<<endl;


    nvalues.clear();
}

void get_hogdescriptor_visu(cv::Mat img,double *descriptorValues, double zoomFac, double angle, int nbins)
{
    cv::Mat color_origImg, origImg, visu;

    img.copyTo(origImg);

    resize(origImg, origImg, *Window_size);
    if(origImg.type()==CV_8UC1)
        cvtColor(origImg, color_origImg, CV_GRAY2RGB);
    else
        origImg.copyTo(color_origImg);

    cvNamedWindow("HOG");
    imshow("HOG", color_origImg);
    cv::waitKey(0);
    cvDestroyWindow("HOG");

    resize(color_origImg, visu, Size(color_origImg.cols*zoomFac, color_origImg.rows*zoomFac));

    int blockSize   = Block_size->width*Block_size->height;
    int cellSize_x  = Cell_size->width;
    int cellSize_y  =Cell_size->height;
    int gradientBinSize = nbins;
    double radRangeForOneBin = angle/(double)gradientBinSize;

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = Window_size->width/Cell_size->width;
    int cells_in_y_dir = Window_size->height/Cell_size->height;
    int totalnrofcells = cells_in_x_dir*cells_in_y_dir;

    double*** gradientStrengths = new double**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];

    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new double*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new double[gradientBinSize];
            cellUpdateCounter[y][x] = 0;

            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;

    for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
    {
        for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
        {
            int cellx = blockx;
            int celly = blocky;
            for (int cellNr=0; cellNr<blockSize; cellNr++)
            {
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    double gradientStrength = descriptorValues[descriptorDataIdx];
                    descriptorDataIdx++;
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
                }
                cellUpdateCounter[celly][cellx]++;

                if((cellNr+1)%Block_size->height==0)
                    celly++;
                 if((cellNr+1)%Block_size->width!=0)
                    cellx++;
                 else
                    cellx=blockx;

            } // for (all cells)
        } // for (all block x pos)
    } // for (all block y pos)


    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            double NrUpdatesForThisCell = (double)cellUpdateCounter[celly][cellx];
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize_x;
            int drawY = celly * cellSize_y;

            int mx = drawX + cellSize_x/2;
            int my = drawY + cellSize_y/2;

            rectangle(visu, Point(drawX*zoomFac,drawY*zoomFac), Point((drawX+cellSize_y)*zoomFac,(drawY+cellSize_x)*zoomFac), CV_RGB(100,100,100), 1);

            for (int bin=0; bin<gradientBinSize; bin++)
            {
                double currentGradStrength = gradientStrengths[celly][cellx][bin];
                if (currentGradStrength==0)
                    continue;

                double currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
//                /cout<<currRad<<endl;
                double dirVecX = cos( currRad );
//                cout<<"cos do rad: "<<dirVecX<<endl;
                double dirVecY = sin( currRad );
//                cout<<"sin do rad: "<<dirVecY<<endl;
                double maxVecLen = cellSize_x/2;
//                cout<<"Strenght: "<<currentGradStrength<<endl;
                double scale = 5; // just a visualization scale, to see the lines better
//                getchar();
                // compute line coordinates
                double x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                double y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                double x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                double y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
//                cout<<"x1 , y1 "<<x1<<" , "<<y1<<endl;
//                cout<<"x2 , y2 "<<x2<<" , "<<y2<<endl;
                // draw gradient visualization
                line(visu, Point(x1*zoomFac,y1*zoomFac), Point(x2*zoomFac,y2*zoomFac), CV_RGB(0,255,0), 1);
            } // for (all bins)

        } // for (cellx)
    } // for (celly)


    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
      for (int x=0; x<cells_in_x_dir; x++)
      {
           delete[] gradientStrengths[y][x];
      }
      delete[] gradientStrengths[y];
      delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;

    cvNamedWindow("HOG");
    imshow("HOG", visu);
    cv::waitKey(0);
    cvDestroyWindow("HOG");
}

bool sortbyPertinencia(const HOG_Result &a, const HOG_Result &b)
{
    if(a.score>b.score)
        return true;
    else
        return false;
}
void IHog_MultiDetect(char *model_file, int n_scale, double *scale)
{

    Size trainingPadding = Size(0, 0);
    int nh,nv;
    nh=Block_size->width*Cell_size->width/2;
    nv=Block_size->height*Cell_size->height/2;

    svm.load(model_file);

    HOGNMS suppression;
    vector<string> dados_placas;


    dados_placas.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-UFOP-gate/101TESTE/rot_vlp/Dados_placas.txt");

    dir_limiares.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/Limiares/Limiar_0.50/DCAM_00");
    dir_limiares.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/Limiares/Limiar_0.60/DCAM_00");
    dir_limiares.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/Limiares/Limiar_0.70/DCAM_00");
    dir_limiares.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/Limiares/Limiar_0.80/DCAM_00");
    dir_limiares.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/Limiares/Limiar_0.85/DCAM_00");
    dir_limiares.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/Limiares/Limiar_0.90/DCAM_00");
    dir_limiares.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/Limiares/Limiar_0.95/DCAM_00");
    Carregar_Dados(dados_placas);
    int t_samples=n_images;

    vector<Rect> Rect_of_interest;


    //tp=new double* [t_samples];
    //fp=new double* [t_samples];
    meu_thresh= new double[n_thresholds];
//    double min_t=-2.0, max_t=2.0;

    meu_thresh[0]=0.2;
//    meu_thresh[0]=min_t;
//    for(int i=1;i<n_thresholds;i++)
//    {
//            meu_thresh[i]= (max_t-min_t)/(n_thresholds-1) + meu_thresh[i-1];
//    }
    pthread_t minhas_threads[number_of_threads];

    time_t t1,t2;
    for (unsigned long currentFile = 0; currentFile < t_samples; ++currentFile)
    {

            Dados *meus_dados=new Dados[number_of_threads];

            const string currentImageFile=(currentFile < positive_example.size() ? positive_example.at(currentFile) : negative_example.at(currentFile - positive_example.size()));
            t1=time(NULL);

            int p_windows=0;
            for(int i=0;i<number_of_threads;i++)
            {
                meus_dados[i].endereco=currentImageFile;
                meus_dados[i].scale=i;
                meus_dados[i].currentFile=currentFile;
                meus_dados[i].scales=&scale_multi[0];
                meus_dados[i].stride=global_stride;
                meus_dados[i].nwindows=0;
                pthread_create(&minhas_threads[i], NULL, Processa_Escala,static_cast<void*>(&meus_dados[i]));
            }
            for(int i=0;i<number_of_threads;i++)
                pthread_join(minhas_threads[i],NULL);

            vector<HOG_Result> meus_roi;
            for(int i=0;i<number_of_threads;i++)
            {
                cout<<"i: "<<i<<" num. janelas: "<<meus_dados[i].nwindows<<endl;
                //total_janelas+=meus_dados[i].nwindows;
                for(int j=0;j<meus_dados[i].meus_roi.size();j++)
                {
                    meus_roi.push_back(meus_dados[i].meus_roi.at(j));
                }
            }

            Avaliar(&meus_roi, currentImageFile, currentFile,meus_roi.size(),&p_windows);
            std::sort(meus_roi.begin(), meus_roi.end(), sortbyPertinencia);
            //vector<HOG_Result> meus_resultados=suppression.ComputeNMSResults(*Window_size,roi_positivos);

            //tp[currentFile]=new double[n_thresholds];
            //fp[currentFile]=new double[n_thresholds];

            fstream det;
            char aux[3];
            Dados_Limiar *n_dados;
            n_dados=new Dados_Limiar[n_limiares];
            pthread_t threads[n_limiares];

            for(int n=0;n<n_limiares;n++)
            {
                itoa(n,aux,10);
                string r="det_curve_";
                r=r+aux+".txt";
                cout<<r<<endl;
                n_dados[n].currentFile=currentFile;
                n_dados[n].nome_file=r;
                n_dados[n].n_roi.reserve(meus_roi.size());
                n_dados[n].n_roi=meus_roi;
                n_dados[n].limiar=n;

                pthread_create(&threads[n], NULL, Processa_Limiar,static_cast<void*>(&n_dados[n]));
//                det.open(r.c_str(),ios::out|ios::app);
//                for(int i=0;i<n_thresholds; i++)
//                {
//                    tp[currentFile][i]=0;
//                    fp[currentFile][i]=0;
//
//                    int stop=-1;
//                    int detected=0;//antes da supressao, quantas regioes que consegui pegar que são real positivas.
//                    for( int j=0; j< meus_roi.size(); j++)
//                    {
//                        if(meu_thresh[i] > (meus_roi.at(j)).score)
//                        {
//                            stop=j;
//                            break;
//                        }
//
//                    }
//                    Supression(stop, &meus_roi, &detected);
//                   // vector<Rect> final;
//
//                    for(int n=0; n<= stop; n++ )
//                    {
//                        if( meus_roi.at(n).sup==false)
//                        {
//                            if( meus_roi.at(n).label[n]==1)
//                            {
//                                tp[currentFile][i]++;
//                            }
//
//                            else
//                                fp[currentFile][i]++;
//                        }
//                        //todos.push_back(meus_roi.at(n).roi);
//                    }
//
//                }
//
//                for(int i=0;i<n_thresholds;i++)
//                {
//                    det<<tp[currentFile][i]<<" "<<fp[currentFile][i]<<endl;
//                }
//                det.close();
//            }
            }
            for(int i=0; i<n_limiares;i++)
                pthread_join(threads[i],NULL);

            t2=time(NULL);
            cout<<"Tempo: "<<t2-t1<<endl;
            meus_roi.clear();
            for(int i=0;i<number_of_threads;i++)
            {
                meus_dados[i].img.release();
                meus_dados[i].meus_roi.clear();
                meus_dados[i].n_values.clear();
            }
            delete meus_dados;
            delete []n_dados;
             janela_out<<"Tempo : "<<t2-t1<<endl;
//             getchar();
            total_janelas=0;
    }

    delete scale_multi;

    //delete []tp;
    //delete []fp;
}

void NormalizarFeatures(double ***meus_hog, char *save_max, char *save_min)
{
    double *max, *min;
    max=new double[getNumberofFeatures()];
    min=new double[getNumberofFeatures()];

    for(int i=0; i< n_images; i++)
    {
        for(int j=0; j<getNumberofFeatures();j++)
        {
            if(*meus_hog[i][j]> max[j])
                max[j]= *meus_hog[i][j];
            if(*meus_hog[i][j]< min[j])
                min[j]= *meus_hog[i][j];
        }
    }
    cout<<"here"<<endl;
    SaveVector(save_max, max);
    SaveVector(save_min, min);
    for(int i=0; i< n_images; i++)
    {
        for(int j=0; j<getNumberofFeatures();j++)
        {
            *meus_hog[i][j]=  2*((*meus_hog[i][j]-min[j])/(max[j]-min[j])) -1 ;
        }
    }
}

void SaveVector(char *name_file, double v[])
{
    fstream faux;
    faux.open(name_file, ios::out);

    for(int i=0;i<getNumberofFeatures();i++)
        faux<<v[i]<<" ";
    faux.close();
}

void Supression(int ind, vector<HOG_Result> *meus_roi, int *detected)
{
    for(int i=0; i<= ind; i++)
        meus_roi->at(i).sup=false;

    for(int i=0; i<= ind; i++)
    {
          if(meus_roi->at(i).sup==false)
          {
            for(int j=i+1; j<=ind; j++)//analiso apenas com relação as de score menor, uma vez que meu vetor já está ordenado
            {
                double intersecao=Area((meus_roi->at(i)).roi & (meus_roi->at(j)).roi );
                double uniao= Area((meus_roi->at(i)).roi)+Area((meus_roi->at(j)).roi)-intersecao;
                if(intersecao/uniao >= 0.50)
                    meus_roi->at(j).sup=true;
            }
          }
    }
}
void* Processa_Escala(void * arg)
{
                //cout<<"Escala..."<<endl;
                //Dados *dados_level=new Dados();
                Dados *dados_level= (static_cast<Dados*>(arg));

                int level=dados_level->scale;
                Thog *meu_hog=new Thog(9,1.0,window_l,window_c,0.02,180,1,false,true,true);
                meu_hog->setWindow_size(Size(window_c,window_l));
                meu_hog->setBlock(Size(tamanho_blocox,tamanho_blocoy));
                meu_hog->setCellSize(Size(tamanho_cellx,tamanho_celly));
                dados_level->img=imread(dados_level->endereco,0);
                int w = dados_level->img.cols*dados_level->scales[level];
                int h = dados_level->img.rows*dados_level->scales[level];

                cv::Mat scaled_image=cv::Mat(h,w,CV_64F);
                //cv::Mat scaled_image;
                //pthread_mutex_lock(&mutex_for_some_value);
                cv::resize(dados_level->img,scaled_image,scaled_image.size());
                dados_level->img.release();
                //pthread_mutex_unlock(&mutex_for_some_value);
                meu_hog->CalculateIntegralHOG(scaled_image,false);

                cout<<"Fim Integral..."<<endl;
                for (int i = Window_size->width; i < w + 1; i += dados_level->stride)
                {
                    for (int j = Window_size->height; j < h + 1; j += dados_level->stride)
                    {
                           //cout<<"( "<<i<<" , "<<j<<" )"<<endl;
                           HOG_Result *n_regiao=new HOG_Result;
                           cv::Mat test_data=cv::Mat(1, getNumberofFeatures(), CV_32F);
                           double *atributos=new double[getNumberofFeatures()];
                           Rect sub_image(i- Window_size->width,j - Window_size->height,Window_size->width,Window_size->height);
                           //calculando o descritor HOG com base na minha implementação.

                           meu_hog->calculateHOG_window(sub_image,atributos);
                           for(int x=0;x<getNumberofFeatures();x++)
                           {
                                test_data.at<float>(0,x)=atributos[x];
                           }

                           //pthread_mutex_lock(&mutex_for_some_value);
                           dados_level->n_values.push_back(svm.predict((test_data),true));
                           n_regiao->score=dados_level->n_values.back();
                           //pthread_mutex_unlock(&mutex_for_some_value);

                           sub_image.x=(sub_image.x+dx)/dados_level->scales[level];
                           sub_image.y=(sub_image.y+dy)/dados_level->scales[level];
                           sub_image.width= (sub_image.width-2*dx)/dados_level->scales[level];
                           sub_image.height= (sub_image.height-2*dy)/dados_level->scales[level];

                           n_regiao->scale=dados_level->scales[level];
                           n_regiao->roi=sub_image;
                           n_regiao->sup=false;

                           //pthread_mutex_lock(&mutex_for_some_value);
                           dados_level->meus_roi.push_back(*n_regiao);
                           //pthread_mutex_unlock(&mutex_for_some_value);
                           dados_level->nwindows=dados_level->nwindows+1;

                           delete []atributos;
                           test_data.release();
                           total_janelas++;
                           delete n_regiao;
                           //cout<<"Calculate HOG para janela: "<<total_janelas<<endl;
                    }
                }
                cout<<"Fim level "<<level<<endl;
                scaled_image.release();
                delete meu_hog;
                //delete dados_level;
                pthread_exit(NULL);
}

void Processa_Base()
{
    vector<string> pos_base_dir;

    static vector<string> ext_valida;
    ext_valida.push_back("png");
    ext_valida.push_back("jpg");
    pos_base_dir.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/imgs-greek-vehicles/");
    int size_dir= pos_base_dir.size();
    vector<string> original_pos;
    files_by_dir2=new int[size_dir];
    getFilesInDirectory(pos_base_dir,original_pos, ext_valida,files_by_dir2);
    for(int i=0;i<size_dir;i++)
            cout<<files_by_dir2[i]<<endl;
    int dir_current=0;
    int indice_by_dir=0;
    char c_aux[3];
    fstream fout;
    string dados= pos_base_dir.at(0) + "rot_vlp/Dados_placas.txt";
    fout.open(dados.c_str(),ios::out);
    for (unsigned long currentFile = 0; currentFile < n_images; ++currentFile)
    {

            const string currentImageFile= positive_example.at(currentFile);//imagem cropped
            const string originalFile= original_pos.at(currentFile);//image original

            cv::Mat n_image=imread(currentImageFile,1);
            cv::Mat original= imread(originalFile,1);
            cv::Mat roi;
            vector<Mat> channels;
            split(n_image, channels);
            int dilation_size=2;
            for(int i=0; i<(channels.size()-1);i++)
            {
                threshold(channels.at(i), channels.at(i), 254, 255, 0);
                int dilation_type;
                dilation_type = MORPH_RECT;
                Mat element = getStructuringElement( dilation_type,
                Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                Point( dilation_size, dilation_size ) );
                /// Apply the dilation operation
                dilate( channels.at(i), channels.at(i), element );
            }
            roi= channels.at(0) + channels.at(1);

            for(int i=0;i<channels.size();i++)
                channels.at(i).release();

            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            findContours( roi, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_KCOS , Point(0, 0) );



            roi.release();

            int area=100;
            int idx=-1;
            for(int i=0;i<contours.size();i++)
            {
                cout<<contours[i].size()<<endl;
                if(contours[i].size()<area && contours[i].size()>=4)//para a outra base (base do pedro é 4)
                {
                    area=contours[i].size();
                    cout<<area<<endl;
                    idx=i;
                }
            }
            Rect meu_roi= boundingRect(contours[idx]);
            Rect vlp_back=boundingRect(contours[idx]);

            Point pt1,pt2;
            pt1.x= meu_roi.x;
            pt1.y=meu_roi.y;
            pt2.x= pt1.x + meu_roi.width;
            pt2.y=pt1.y+meu_roi.height;

            vlp_back.x=vlp_back.x-dx;
            vlp_back.y =vlp_back.y -dy;
            vlp_back.width= vlp_back.width+2*dx;
            vlp_back.height= vlp_back.height+2*dy;

            cv::Mat subimage(original,meu_roi);
            cv::Mat back_image(original,vlp_back);
            if(indice_by_dir<files_by_dir2[dir_current])
            {
                fout<<meu_roi.x<<" , "<<meu_roi.y<<" , "<<meu_roi.width<<" , "<<meu_roi.height<<endl;
                string results= pos_base_dir.at(dir_current)+ "rot_vlp/lp/DCAM_00";
                string results_back= pos_base_dir.at(dir_current)+ "rot_vlp/positivas/DCAM_00";
                itoa(indice_by_dir,c_aux,10);
                imwrite(results+ c_aux +".jpg",subimage);
                imwrite(results_back+ c_aux +".jpg",back_image);
                indice_by_dir++;

            }
            else
            {
                indice_by_dir=0;
                dir_current++;


                string results= pos_base_dir.at(dir_current)+ "rot_vlp/lp/DCAM_00";
                string results_back= pos_base_dir.at(dir_current)+ "rot_vlp/positivas/DCAM_00";
                itoa(indice_by_dir,c_aux,10);
                imwrite(results+ c_aux +".jpg",subimage);
                imwrite(results_back+ c_aux +".jpg",back_image);
                indice_by_dir++;

                string dados= pos_base_dir.at(dir_current) + "rot_vlp/Dados_placas.txt";
                fout.close();
                fout.open(dados.c_str(),ios::out);
                fout<<meu_roi.x<<" , "<<meu_roi.y<<" , "<<meu_roi.width<<" , "<<meu_roi.height<<endl;
            }

              /// Show in a window
//

              subimage.release();
              back_image.release();
    }
}
int current_image=0;
void * Processa_Limiar(void * arg)
{

                Dados_Limiar *dados_limiar;
                dados_limiar= (static_cast<Dados_Limiar*>(arg));
                //cout<<dados_limiar->limiar<<endl;
                fstream det;
                det.open(dados_limiar->nome_file.c_str(),ios::out|ios::app);
                string currentImageFile=positive_example.at(dados_limiar->currentFile);

                int *ntp=new int[n_thresholds];
                int *nfp=new int[n_thresholds] ;
                vector<Rect> roi_tp, roi_fp, roi_window;
                vector<float> score_tp;
                cv::Mat img_print=imread(currentImageFile,1);
                Rect *gt,max_rect;

                for(int i=0;i<n_thresholds; i++)
                {
                    double max=0;
                    cout<<"Novo Threshold, favor renomear as pastas..."<<endl;

                    ntp[i]=0;
                    nfp[i]=0;
                    int stop=-1;
                    int detected=0;//antes da supressao, quantas regioes que consegui pegar que são real positivas.
                    for( int j=0; j< dados_limiar->n_roi.size(); j++)
                    {
                        if(meu_thresh[i] > (dados_limiar->n_roi.at(j)).score)
                        {
                            stop=j;
                            break;
                        }

                    }
                    //pthread_mutex_lock(&mutex_for_some_value);
                    Supression(stop, &dados_limiar->n_roi, &detected);
                    gt=new Rect(x.at(dados_limiar->currentFile),y.at(dados_limiar->currentFile),
                        w.at(dados_limiar->currentFile), h.at(dados_limiar->currentFile));
                    //gt=new Rect(x.back(),y.back(),w.back(), h.back());

                    for(int n=0; n<= stop; n++ )
                    {

                        if( dados_limiar->n_roi.at(n).sup==false)
                        {

                              roi_window.push_back(Rect(dados_limiar->n_roi.at(n).roi.x,
                                                   dados_limiar->n_roi.at(n).roi.y,
                                                        dados_limiar->n_roi.at(n).roi.width,
                                                        dados_limiar->n_roi.at(n).roi.height));


                            if(dados_limiar->n_roi.at(n).label[dados_limiar->limiar]==1)
                            {
                                        ntp[i]++;
                                        HOG_Result best_result;
                                        best_result.roi = dados_limiar->n_roi.at(n).roi;

                                        double intersecao=Area(*gt&best_result.roi);
                                        double uniao= Area(*gt)+Area(best_result.roi)- intersecao;
                                        double int_unio= intersecao/uniao;
                                        cout<<"Intersecao/ Uniao :"<<int_unio<<endl;
            //
                                        if(int_unio>max)
                                        {
                                            max=int_unio;
                                            max_rect=best_result.roi;

                                        }
                             }
                             else
                                nfp[i]++;

                        }
                    }
                  cv::Mat n_image=imread(currentImageFile,1);

                  rectangle(img_print, gt->tl(), gt->br(), Scalar(0,0,255),4);
                  showDetections(roi_window,img_print,0);

                  rectangle(n_image, gt->tl(), gt->br(), Scalar(0,0,255),4);
                  showDetections(roi_tp,n_image,0);

                  char c_aux[3];
                  itoa(current_image,c_aux,10);
                  string file=dir_limiares.at(dados_limiar->limiar)+string(c_aux)+".jpg";
                  string file2=dir_limiares.at(dados_limiar->limiar+1)+string(c_aux)+".jpg";
                  imwrite(file,img_print);
                  imwrite(file2,n_image);
                  current_image++;
                  det<<ntp[i]<<" "<<nfp[i]<<" "<<max<<endl;

                  n_image.release();

                }

                img_print.release();
                roi_tp.clear();
                roi_window.clear();

                det.close();
                delete []ntp;
                delete []nfp;
                pthread_exit(NULL);
}
void TestePedro()
{
    int t_samples=n_images;

    vector<string> dados_placas;


    dados_placas.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/B/Teste_00033_xvid/Resultados/Dados_placas.txt");
    Carregar_Dados(dados_placas);
    vector<double> r_x, r_y, r_w, r_h;
    r_x=x; r_y=y; r_w=w; r_h=h;
    x.clear(), y.clear(), w.clear(), h.clear();
    dados_placas.clear();

    dados_placas.push_back("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/B/Teste_00033_xvid/rot_vlp/Dados_placas.txt");
    Carregar_Dados(dados_placas);
    fstream det;
    det.open("C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/B/Teste_00033_xvid/Resultados/RESULTADOS_PEDRO/pedro_curve.txt",ios::out);
    cout<<t_samples<<endl;
    cout<<r_x.size()<<endl;
    cout<<x.size()<<endl;
    for (unsigned long currentFile = 0; currentFile < t_samples; ++currentFile)
    {
            const string currentImageFile=(currentFile < positive_example.size() ? positive_example.at(currentFile) : negative_example.at(currentFile - positive_example.size()));
            cout<<currentImageFile<<endl;
            cv::Mat img=imread(currentImageFile,0);
//			Plate p;
//			p.vlpl(img);
			Rect *gt;
			if (r_x.at(currentFile)>0)
			{
			    Point p1,p2;
			    Rect detection(r_x.at(currentFile),r_y.at(currentFile), r_w.at(currentFile), r_h.at(currentFile));
			    gt=new Rect(x.at(currentFile),y.at(currentFile),
                        w.at(currentFile), h.at(currentFile));
                cvtColor(img,img,CV_GRAY2RGB);

                rectangle(img, gt->tl(), gt->br(), Scalar(180,105, 255), 1);
			    rectangle(img, detection.tl(), detection.br(), Scalar(255,0,0), 1);

                double intersecao=Area(*gt&detection);
                double uniao= Area(*gt)+Area(detection)- intersecao;
                det<<intersecao/uniao<<endl;
                char c_aux[3];
                itoa(currentFile,c_aux,10);
                string file="C:/Users/Gustavo/Desktop/Raphael/HOG/imgs-LPIV/B/Teste_00033_xvid/Resultados/RESULTADOS_PEDRO/DCAM_0"+string(c_aux)+".jpg";
                imwrite(file,img);
			}
			img.release();
    }
}
