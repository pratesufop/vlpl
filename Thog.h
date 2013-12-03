#ifndef THOG_H
#define THOG_H

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

using namespace std;
using namespace cv;


class Thog
{
  private:
    int ddepth;
    double scale;
    int delta;
    int nbins;
    vector<cv::Mat> integrals;
    double angle;
    double eps;
    int n_images;
    Size block_size;
    Size window_size;
    Size cell_size;

    bool emprega_normal2;
    fstream faux;
    int n_features;
    bool emprega_normal1;

    bool BoW_on;

    int totalcenters;

  public:
    Thog(int n_b, double s, int n_rows, int n_cols, double neps,double n_angle, int n_scales, bool L2, bool L1, bool bow_on);
    ~Thog();
    void CalculateIntegralHOG(cv::Mat image_raw,bool sinal);
    //Calcula o integral histogram para cada um dos bins da imagem, armazenando em integrals
    void CalculaHOG_Retangulo(cv::Rect cell,double *hog_cell,double *sum);
    //Método empregado para o cálculo do HOG de cada célula, sendo a célula passada como um cv::Rect e a resposta é dada por cv::Mat com nove posições
    //onde cada posição corresponde a um bin, ou seja, uma faixa angular.
    void setBlock(Size n_bsize);
    //altero o tamanho dos blocos.
    void setWindow_size(Size w);
    //altero o tamanho da janela de detecção.
    Size getBlockSize();
    // retorno o tamanho dos blocos, o tamanho dos blocos é definido como sendo a quantidade de células,
    //por exemplo, um bloco com tamanho h= 2 e w=1 tem duas células na vertical e uma na horizontal
    Size getWindowSize();
    // retorno o tamanho da janela de detecção.
    Size getCellSize();
    //retorno o tamanho das células em número de pixels.
    int getNumberFeatures();
    //considera o overlapping de 50%, retorna o número de células vezes o número de bins por célula
    //o número é dado pela divisão do (tamanho da janela -  tamanho bloco)/tamanho da célula e soma mais 1.
    //uma modificação do foi acrescentada de forma a alterar o número de features quando realizo a normalização da
    //imagem.
    void setNumberFeatures(int n);
    //conseguiu determinar o número de features.
    int getNumberSamples();
    //retorna o número de imagens usadas. Um soma das positivas mais negativas.
    int getCell_Block();
    //retorna apenas o número de células por  bloco wxh
    void calculateHOG_window(cv::Rect window, double *window_feature_vector);
    //dada uma janela, divide-se ela em blocos e para cada bloco calcula-se o vetor de features por bloco,
    //cujo tamanho é dado por número de células por bloco vezes o número de bins por célula!
    void train(bool cross_validation, int n_folds, char *savexml,bool use_bow, bool supervisionado, bool bow_cell, bool normalizar);
    //No treinamento é feita a varredura apenas naquele tamanho de janela, por isso, apenas se chama a função que
    //calcula o vetor de features para uma janela.
    void setCellSize(Size ncell);
    void calculateHOG_block(cv::Rect block, double *hog_block,bool normalizar_block);
    //calcula-se o vetor de features por bloco, chamando a função calcula Hog por retângulo responsável por
    //calcular o vetor de HOG por célula. Ao final o vetor é normalizado dividindo-se cada elemento pela raiz(soma² + tamanho do vetor).
    void trainSVM(char *savexml,char *pos_file, char *neg_file);
    //Esta função carrega o vetor de features gravado para realizar o treinamento do SVM. Para isto cria duas matrizes para armazenar os exemplos
    //positivos e negativos, cada linha corresponde a um exemplo e cada coluna a um feature. Além disso, temos que passar uma matriz com os trainClasses
    // que no caso é 1 para os exemplos positivose e  2 para os negativos.
    int getPositive();
    double Exp(int n, int k, double z);
    double Gaussian(double z, double mu, double sigma);
    double getWx(int x);
    double getWy(int y);
    cv::Mat MyDetection(cv::Mat sample,int n_scale, int n_step, bool normalizar);
    void getPiramide(cv::Mat sample, double *scale, int n_scale);
    void LoadSVM(char *loadxml);
    vector<Rect> rank(vector<double>& image, vector<Point>& p_candidates);
    //Função que realiza um rank dos votos, sabendo que queremos os 100 melhores, por exemplo, ele verifica
    //se a posição no vetor que tenho é melhor  que uma das últimas posições do vetor e maior que o limiar
    //caso afirmativo, aceito esta solução e armazeno aquele ponto. Pois, sabendo ponto e o width e heigth
    //já consigo montar o retângulo.
    void showDetections(const vector<Rect>& found, Mat& imageData);
    void addImage(cv::Mat aux);
    //Em add image estou fazendo duas cópias da imagem de entrada de forma que seja possível calcular a normalização
    //da sem alterar o valor da imagem original.
    void compute_binomial_weights (double *weight, int rwt);
    void compute_gaussian_weights(double *weight, int rwt);
    double get_grey_dev(double AVG, double noise, Rect region,double **p, double n_sum);
    double get_grey_avg(Rect region,double **p, double n_sum);
    void normalize_grey_image (int w, int h, int x_rwt, int y_rwt, double noise, double **p, double n_sum);
    double** getProduct(double  *a,double *b, int nx,int ny, double *sum);
    void get_hogdescriptor_visu(cv::Mat img,vector<float>& descriptorValues, double zoomFac);
    //Função empregada para visualizar os gradientes.
    void DallalHog(bool cross_validation , int n_folds,char *savexml,bool use_bow, bool supervisionado, bool bow_cell);
    //Função que utiliza da implementação do HOG proposta por Dallal and Triggs.
    //cross validation igual a true ativa o modo de treino automático, utilizando o parâmetro n_folds.
    //O parâmetro savexml informa qual é o arquivo onde ficaram salvos os dados do classificador SVM.
    // use_bow é um variável booleana que indica se desejamos ou não utilizar a representação em bag of features.
    // Se bow for usado, temos que definir o que é uma instância: se bow_cell, cada célula é uma instância.
    //Se for supervisionado, indica que apenas os exemplos da classe positica serão empregados para extrair os features do BoW.
    void calculateFeaturesFromInput(const string& imageFilename, vector<float>& featureVector, bool dallal_hog);
    void detectTest(Mat& imageData);
    void NormalizarFeature(char *n_train,const char *file_norm );
    void RealizarTeste(char *n_teste, string results, char *tests, bool use_bow, bool bow_cell, bool validacao, bool dallal_hog);
    //n_teste é o nome do arquivo de onde se quer carregar o modelo do classificador SVM.
    //bool validacao define de qual dos diretórios serão extraídas as imagens para realizar os testes. Se validacao for
    //igual a true, então serão analisados do conjunto treino_positivas e treino_negativas.
    //string results traz o endereço do diretório de saída, enquanto que o diretório tests o de entrada.
    cv::Mat Bag_of_Features(cv::Mat data, bool save_hog, double p_sample, int ncenters);
    void Save_BoW(cv::Mat BoW_data);
    void Save_BoW(cv::Mat BoW_data, int indice);
    void SaveVector(int t, char *name_file, double v[]);
    void SaveVector(int t, char *name_file, float v[]);
    double* GetVector(char *name_file, int t);
    void NormalizarTeste(double *v);
    vector<float> loadSVMfromModelFile(const char* filename);
    void MultiScaleDetection(char *model_file, char *dir);
    void setSVMTrainAutoParams( CvParamGrid& c_grid, CvParamGrid& gamma_grid,
                            CvParamGrid& p_grid, CvParamGrid& nu_grid,
                            CvParamGrid& coef_grid, CvParamGrid& degree_grid );
    void MyMultiDetection(char *dir,string results, char *model_file, int n_scale, int n_step, bool normalizar, bool use_bow, bool bow_cell);
    //Neste MultiDetection uso o compute da implementação do Dallal.
    void IHog_MultiDetect(char *dir_in, string results, char *model_file, int n_scale, int n_step, bool normalizar, bool use_bow, bool bow_cell);
    //Trata-se de uma detecção em multiscale que faz uso do Integral Histogram implementado atravás da função CalculateHOGWindow, diferente da outra
    //implementação multidetection que faz uso do compute HOG da implementação do Dalal.
    //model_file indica qual o modelo de entrada do classificador SVM será utilizado.
    //n_scale indica o número de escalas que está sendo usada.
    //n_step indica o tamanho do passo entre duas janelas.
    //normalizar indica se será necessário realizar a normalização.
};
#endif


