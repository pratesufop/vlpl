#include "Thog.h"


Thog::Thog(int n_b, double s, int n_rows, int n_cols, double neps, double n_angle,int n_scale, bool L2, bool L1, bool bow_on)
{
    ddepth=CV_64F;
    scale=s;
    delta=0;
    nbins=n_b;
    eps=neps;
    angle=n_angle;
    n_images=0;
    emprega_normal1=L1;
    n_features=0;
    BoW_on=bow_on;
    totalcenters=10;

}

Thog::~Thog()
{
    for (int i=0;i<nbins;i++)
    {
            integrals.at(i).release();
    }
}

void Thog::CalculateIntegralHOG(cv::Mat image_raw, bool sinal)
{
    cout<<"Integral Image"<<endl;
    if(image_raw.type()!=CV_8UC1)
        cvtColor(image_raw,image_raw,CV_RGB2GRAY);
    int img_r, img_c;
    img_r= image_raw.rows;
    img_c=image_raw.cols;

    cout<<"sobel"<<endl;
    cv::Mat xsobel,ysobel;
    xsobel=cv::Mat(cv::Mat::zeros(img_r,img_c,CV_64F));
    ysobel=cv::Mat(cv::Mat::zeros(img_r,img_c,CV_64F));

    Sobel(image_raw,xsobel,ddepth,1,0,1,scale,delta,BORDER_REPLICATE);
    Sobel(image_raw,ysobel,ddepth,0,1,1,scale,delta,BORDER_REPLICATE);
    image_raw.release();

    //Criando uma imagem para cada bin
    cv::Mat bins[nbins];
    double *faixa_angles=new double[nbins];
    cout<<"bins"<<endl;
    for (int i=0;i<nbins;i++)
    {
        bins[i]=cv::Mat(cv::Mat::zeros(img_r,img_c,ddepth));
        faixa_angles[i]=(angle/nbins)*(i+1);
    }

    cout<<"gradients"<<endl;
    double px,py,temp_angulo, temp_magnitude;
    double epsilon=0.00000001;
    for(int y=0;y<img_r;y++)
    {
        double *ptr1=xsobel.ptr<double>(y);
        double *ptr2=ysobel.ptr<double>(y);
        double **ptr3=(double**) malloc(nbins * sizeof(double*));

        for(int n=0;n<nbins;n++)
        {
            ptr3[n]=bins[n].ptr<double>(y);//cada ptr3 aponta para um vetor de bins
        }
        for(int x=0;x<img_c;x++)
        {
            px=ptr1[x];
            py=ptr2[x];
            for(int i=0;i<nbins;i++)
            {
                ptr3[i][x]=0.0;
            }
            if(px==0)
                temp_angulo=floor((std::atan2(py,(px+epsilon))));
            else
                temp_angulo=floor(((std::atan2(py,px))*180/3.14));

            if(sinal)
            {
                if(temp_angulo<0)
                    temp_angulo+=360;
            }
            else
            {
                if(temp_angulo<0)
                    temp_angulo+=180;
            }
            temp_magnitude=sqrt(px*px + py*py);
            if(temp_magnitude<eps)
                temp_magnitude=0;

            for(int i=0;i<nbins;i++)
            {
                if(temp_angulo<=faixa_angles[i])
                {
                    //decompor o vetor de gradiente em dois planos.
                    ptr3[i][x]=temp_magnitude;
                    //cout<<"angle: "<<faixa_angles[i]<<" "<<"nmag: "<<(double)ptr3[i][x]<<endl;
                    break;
                }
            }
        }
        free(ptr3);
        //ptr1=NULL;
        //ptr2=NULL;
    }
    delete []faixa_angles;
    xsobel.release();
    ysobel.release();
    cout<<"integrals"<<endl;
    //integrals=new cv::Mat*[nbins];
    for (int i=0;i<nbins;i++)
    {
        integrals.push_back(cv::Mat(cv::Mat::zeros(img_r+1,img_c+1,CV_64F)));
    }

    for(int n=0;n<nbins;n++)
    {
           cv::integral(bins[n],integrals.at(n),CV_64F);
           bins[n].release();

    }
    cout<<"fim"<<endl;

}

void Thog::CalculaHOG_Retangulo(cv::Rect cell,double *hog_cell,double *sum)
{

    double a,b,c,d;
    double *ptr;
    for(int i=0;i<nbins;i++)
    {
        ptr=(integrals.at(i)).ptr<double>(cell.y);
        a=ptr[cell.x];
        ptr=(integrals.at(i)).ptr<double>(cell.y+cell.height);
        b=ptr[cell.x+cell.width];
        ptr=(integrals.at(i)).ptr<double>(cell.y);
        c=ptr[cell.x+cell.width];
        ptr=(integrals.at(i)).ptr<double>(cell.y+cell.height);
        d=ptr[cell.x];
        *sum =*sum + abs(a+b-(c+d));
        hog_cell[i]=(a+b-(c+d));

    }
    //ptr=NULL;
}

void Thog::setBlock(Size n_bsize)
{
    block_size=n_bsize;
}

Size Thog::getBlockSize()
{
    return block_size;
}

void Thog::setWindow_size(Size w)
{
    window_size=w;
}

Size Thog::getWindowSize()
{
    return window_size;
}

Size Thog::getCellSize()
{
    return cell_size;
}

void Thog::setCellSize(Size ncell)
{
    cell_size=ncell;
}

int Thog::getCell_Block()
{
    Size b=getBlockSize();
    return b.width*b.height;
}

 int Thog::getNumberSamples()
 {
     return n_images;
 }

int Thog::getNumberFeatures()
{

    Size nw=getWindowSize();;
    Size ncell=getCellSize();
    Size nblock=getBlockSize();
    int nb_h=((nw.width-ncell.width*nblock.width)/ncell.width)+1;
    int nb_v=((nw.height-ncell.height*nblock.height)/ncell.height)+1;
    n_features=nb_h*nb_v*nbins*getCell_Block();

    return n_features;
}

void Thog::setNumberFeatures(int n)
{
    n_features=n;
}

void Thog::calculateHOG_window(cv::Rect window, double *window_feature_vector)
{
    Size b=getBlockSize();
    Size c=getCellSize();

    int cell_width=c.width;
    int cell_height=c.height;
    int b_width=b.width;
    int b_height=b.height;
    int b_h_pixels=cell_height*b_height;
    int b_w_pixels=cell_width*b_width;

    int block_start_x, block_start_y;


    int cont=0;
    int iw=0;//indice para a janela.
    for(block_start_y = window.y; block_start_y <= window.y + window.height - b_h_pixels;block_start_y += cell_height)
    {
        for(block_start_x = window.x; block_start_x<= window.x + window.width - b_w_pixels;block_start_x += cell_width)
        {
            double *vector_block=new double[getCell_Block()*nbins];
            cv::Rect *nblock=new Rect(block_start_x,block_start_y, b_w_pixels, b_h_pixels);
            calculateHOG_block(*nblock, vector_block,emprega_normal1);
            int it=0;
            while(it<getCell_Block()*nbins)
            {
                window_feature_vector[iw]=vector_block[it];
                it++;
                iw++;
            }
            delete vector_block;
            delete nblock;
        }
    }
}
void Thog::calculateHOG_block(cv::Rect block, double *hog_block, bool normalizar_block)
{
    int cell_start_x, cell_start_y;
    //vector<double> vector_cell;

    Size *cell=new Size();
    *cell= getCellSize();
    double sum=0;
    int ib=0;//índice para as posições do bloco.
    for (cell_start_y = block.y; cell_start_y <=block.y + block.height - cell->height;cell_start_y += cell->height)
    {
        for (cell_start_x = block.x; cell_start_x <=block.x + block.width - cell->width;cell_start_x += cell->width)
        {
            double *vector_cell;
            vector_cell=new double[nbins];
            //cout<<cell_start_x<<" x "<<cell_start_y<<" w: "<<cell.width<<" h: "<<cell.height<<endl;
            cv::Rect *ncell=new Rect(cell_start_x,cell_start_y, cell->width, cell->height);
            //cout<<"cell criada..."<<endl;
            CalculaHOG_Retangulo(*ncell, vector_cell,&sum);
            int it=0;
            while(it<nbins)
            {
                hog_block[ib]=vector_cell[it];
                ib++;
                it++;
            }
            delete vector_cell;
            delete ncell;
        }
    }
    //Realizando a normalização L1 por bloco
    if(normalizar_block)
    {
        int it=0;

        while(it<ib )
        {
            hog_block[it]=hog_block[it]/sum;
            it++;
        }
        //faux<<endl;
    }

    delete cell;
}

double Thog::Exp(int n, int k, double z)
{
		if(n==1)
            return 1;

		double mu = 0.01;
		double sigma = 0.5;
		double avg = -mu + (1 + 2*mu)/(double)(n - 1)*k;
		double dev = sigma/n;
		return Gaussian (z, avg, dev);
}
//
double Thog::Gaussian(double z, double mu, double sigma)
{
		return exp(-((z - mu)*(z - mu))/(2 * sigma * sigma));
}
double Thog::getWy(int y)
{
    int ymax, ymin;
    ymax=(((y+1)/cell_size.height)+1)*cell_size.height;
    ymin=(((y+1)/cell_size.height))*cell_size.height;
    double z=(y+1)%cell_size.height + ((y+1)/cell_size.height)*cell_size.height;

    z=(z-ymin)/(double)(ymax-ymin);
    int cy=block_size.height-(ymax/cell_size.height)%block_size.height;
    double wy=Exp(block_size.height,cy,z);

    return wy;
}
void Thog::showDetections(const vector<Rect>& found, Mat& imageData)
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
        rectangle(imageData, r.tl(), r.br(), Scalar(64, 255, 64), 3);
    }
}


void Thog::setSVMTrainAutoParams( CvParamGrid& c_grid, CvParamGrid& gamma_grid,
                            CvParamGrid& p_grid, CvParamGrid& nu_grid,
                            CvParamGrid& coef_grid, CvParamGrid& degree_grid )
{
    c_grid = CvSVM::get_default_grid(CvSVM::C);

    gamma_grid = CvSVM::get_default_grid(CvSVM::GAMMA);

    p_grid = CvSVM::get_default_grid(CvSVM::P);
    p_grid.step = 0;

    nu_grid = CvSVM::get_default_grid(CvSVM::NU);
    nu_grid.step = 0;

    coef_grid = CvSVM::get_default_grid(CvSVM::COEF);
    coef_grid.step = 0;

    degree_grid = CvSVM::get_default_grid(CvSVM::DEGREE);
    degree_grid.step = 0;
}
