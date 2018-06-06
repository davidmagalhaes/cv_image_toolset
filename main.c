#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <opencv/cv.h>
#include <highgui.h>
#include <stdarg.h>
#include <opencv2/core/core_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc.hpp>
#include "opencv2/opencv.hpp"   
#include "opencv2/ml/ml.hpp"

struct _point{
	int x;
	int y;
};

typedef struct _point TPoint;

struct _binimg{
	unsigned char *data;
	TPoint *borders;
	int borders_size;
	unsigned int* signature;
	int width;
	int height;
	unsigned char val;
	struct _binimg *next;
};

typedef struct _binimg Tbinimg;

struct _stack {
	int *value;
	struct _stack *next;
};

typedef struct _stack TStack;

//Input functions
double **ask_mask(int*, int*);
char ask_limiars(unsigned char**, int*);
void ask_locallim_data(unsigned char*, int*, int*);
FILE *ask_outputfile();
FILE *ask_inputfile(char **filepath);

//Utils
void clean_stdin();
unsigned long file_size(FILE*);
void **create2dArr(int, int, size_t);
void free2dArr(void **arr, int size);
void showImg(const char*, CvMat*);
int min(int a, int b);
int max(int a, int b);
int *pop(TStack *stack);
void push(TStack *stack, int va_length, ...);
int stack_size(TStack *stack);
TStack *new_stack();
void free_stack(TStack *stack);


//Menus
int  menu_loadimage(char **filename, char** bkpfilename); 
void menu_opimg(char *filename, unsigned char *memoryMatrix, int matrix_size_x, int matrix_size_y);
void menu_showimage(char *filename);
void menu_conv(char *filename);
void menu_histogram(char *filename);
void menu_equalize(char *filename);
void menu_limiarize(char *filename);
void menu_median(char *filename);
void menu_reggrowth(char *filename);
void menu_hough_transform(char *filename);
void menu_numbimg();

//Menu Auxiliaries
void confirm_save(char *filename, CvMat*);
int ask_repeatop();
void show_and_trysave(char *filename, int, int, unsigned char*);
CvMat *matFromFile(char *filename);


//Business Functions
unsigned char *convolution(const unsigned char *matrix, double **mask, int matrix_size_x, int matrix_size_y, int mask_size_x, int mask_size_y);
unsigned char *normalize(const unsigned int *data, int data_size);
unsigned int *histogram(const unsigned char *matrix, int matrix_size_x, int matrix_size_y);
void equalize_htgr(unsigned int *histogram, unsigned char *matrix, int matrix_size_x, int matrix_size_y);
unsigned char *limiarize(unsigned char *matrix, unsigned char *pivot, int pivot_sz, int matrix_size_x, int matrix_size_y);
unsigned char *local_limiarize(unsigned char *matrix, unsigned char contrast, int matrix_size_x, int matrix_size_y, int window_size_x, int window_size_y);
unsigned char *reggrowth(unsigned char*, int matrix_size_x, int matrix_size_y, int seed_pos_x, int seed_pos_y);
unsigned char *median3x3(const unsigned char *matrix1, int matrix_size_x, int matrix_size_y);
unsigned char *hough_transform(const unsigned char *matrix, int matrix_size_x, int matrix_size_y);
void show_histogram(unsigned int *histogram);
unsigned char *sum_images(const unsigned char *matrix1, const unsigned char *matrix2, int min_x_m1_m2, int min_y_m1_m2);
unsigned char *radquadsum_images(const unsigned char *matrix1, const unsigned char *matrix2, int min_x_m1_m2, int min_y_m1_m2);
void extract_attr_signature(Tbinimg*, int, int, int);
void classify(cv::Ptr<cv::ml::SVM> svm, Tbinimg *head, int n_img, int imgx, int imgy);
cv::Ptr<cv::ml::SVM> svm_training(Tbinimg *head, int n_img, int imgx, int imgy);

//TODO
//Imagem do histograma: implementar função show_histogram
//Pensar em um nome para a função _conv e integrá-la no projeto


//Input Functions
int main(int argsize, char **args){
	char acao;
	char *filename = (char*) malloc(150);
	int it, jt;
	char *bkpfilename = (char*) malloc(150);
	CvMat *img = NULL;
	CvMat *memoryImg = NULL;
	int memoryImg_width = 0, memoryImg_height = 0;

	strcpy(filename, "");
	strcpy(bkpfilename, "");

	cvNamedWindow( "mainWin", CV_WINDOW_AUTOSIZE ); // Create a window for display.
	cvMoveWindow("mainWin", 100, 100);

	while(acao != 'x'){
		printf("\n\t\t\t Image toolset \n\n");
		printf("\t\tEscolha uma ação: \n");
		printf("\t\t0 - Carregar imagem LENNA.JPG\n");
		printf("\t\t1 - Carregar uma imagem\n");
		printf("\t\t2 - Guardar imagem em memória\n");
		printf("\t\t3 - Operar com imagem em memória\n");
		printf("\t\t4 - Exibir Imagem\n");
		printf("\t\t5 - Exibir Histograma\n");
		printf("\t\t6 - Convolução\n");
		printf("\t\t7 - Equalização\n");
		printf("\t\t8 - Limiariarização\n");
		printf("\t\t9 - Mediana\n");
		printf("\t\ta - Crescimento de Região\n");
		printf("\t\tb - Transformada de Hough\n");
		printf("\t\tc - Testes de extração de atributos\n");
		printf("\t\tx - Sair\n\n");

		printf("Opção: ");
		scanf("%c", &acao);

		switch(acao){
			case '0' : 
				strcpy(filename, "LENNA.JPG"); 
				strcpy(bkpfilename, filename);

				if(img != NULL && img != memoryImg){
					cvReleaseMat(&img);
				}

				img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

				cvShowImage( "mainWin", img ); 
				cvWaitKey(100);
			break;

			case '1' : menu_loadimage(&filename, &bkpfilename); break;

			case '2' : 
				if(strcmp(filename, "")){
					if(memoryImg != NULL){
						cvReleaseMat(&memoryImg);
					}

					memoryImg = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE); 
					memoryImg_width = memoryImg->width; 
					memoryImg_height = memoryImg->height; 
					printf("Imagem guardada com sucesso!\n");
				}
				else{
					printf("ERRO: nenhum arquivo carregado!\n");
				}
			break;

			case '3' : menu_opimg(filename, memoryImg == NULL ? NULL : memoryImg->data.ptr, memoryImg_width, memoryImg_height); break;
			case '4' : menu_showimage(filename);  		break;
			case '5' : menu_histogram(filename);  		break;
			case '6' : menu_conv(filename); 			break;
			case '7' : menu_equalize(filename) ; 		break;
			case '8' : menu_limiarize(filename); 		break;
			case '9' : menu_median(filename);			break;
			case 'a' : menu_reggrowth(filename);		break;
			case 'b' : menu_hough_transform(filename);	break;
			case 'c' : menu_numbimg();					break;
		}

		clean_stdin();
	}

	free(filename);
	free(bkpfilename);

	if(img != NULL){
		cvReleaseMat(&img);
	}
	if(memoryImg != NULL){
		cvReleaseMat(&memoryImg);
	}

	return 0;
} 

int menu_loadimage(char **filename, char **bkpfilename){
	FILE *loadedFile = ask_inputfile(filename);
	int result = loadedFile != NULL;

	free(loadedFile);
	//strcpy(filename, "LENNA.JPG");

	if(result){
		strcpy(*bkpfilename, *filename);
		CvMat *img = cvLoadImageM(*filename, CV_LOAD_IMAGE_GRAYSCALE);

		cvShowImage( "mainWin", img ); 
		cvWaitKey(100);

		cvReleaseMat(&img);
	}
	else{
		printf("Não foi possível carregar o arquivo %s", *filename);
		strcpy(*filename, *bkpfilename);
	}

	return result;
}

void menu_opimg(char *filename, unsigned char *memoryMatrix, int matrix_size_x, int matrix_size_y){
	CvMat *img, resimg;
	int i, j, op;
	unsigned char* sumimg = NULL;

	if(strcmp(filename, "")){
		if(memoryMatrix != NULL){
			img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

			printf("Escolha uma operação: \n");
			printf("\t\t1 - Soma simples das duas imagens\n");
			printf("\t\t2 - Raiz da soma dos quadrados das imagens\n");			

			printf("Opção:");
			scanf("%d", &op);

			switch(op){
				case 1 : sumimg = sum_images(img->data.ptr, memoryMatrix, min(img->width, matrix_size_x), min(img->height, matrix_size_y)); break;
				case 2 : sumimg = radquadsum_images(img->data.ptr, memoryMatrix, min(img->width, matrix_size_x), min(img->height, matrix_size_y)); break;
				default : printf("Opção inválida!\n");
			}

			if(sumimg != NULL){
				resimg = cvMat(min(img->height, matrix_size_y), min(img->width, matrix_size_x), CV_8UC1, sumimg);

				printf("A imagem ficará assim\n");

				cvShowImage( "mainWin", &resimg); 
				cvWaitKey(20);

				confirm_save(filename, &resimg);

				free(sumimg);
			}
			
			cvReleaseMat(&img);
		}
		else{
			printf("ERRO: nenhuma imagem em memória!\n");
		}
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
	}
}

void menu_showimage(char *filename){
	CvMat *img;

	if(img = matFromFile(filename)){
		cvShowImage( "mainWin", img ); 	
		cvWaitKey(100);

		cvReleaseMat(&img);
	}
}

void menu_hough_transform(char* filename){
	CvMat *img;
	unsigned char* newimgdata;

	if(img = matFromFile(filename)){
		newimgdata = hough_transform(img->data.ptr, img->height, img->width);

		show_and_trysave(filename, 180, sqrt(2.0)*max(img->height, img->width), newimgdata);

		free(newimgdata);
	}
}

void menu_numbimg(){
	char * line = NULL;
    size_t len = 0;
    ssize_t read;
	FILE *fp = fopen("ocr_car_numbers_rotulado.txt", "r");
	int n_img = 0;
	int i;
	Tbinimg *head = NULL, *base = NULL;
	int acertos = 0, erros = 0;

	printf("carregando arquivo de números...\n");
	clean_stdin();

	if (fp == NULL)
        exit(EXIT_FAILURE);

    while ((read = getline(&line, &len, fp)) != -1) {
    	Tbinimg *binimg = (Tbinimg*) malloc(sizeof(Tbinimg));
    	
    	binimg->width = 35;
    	binimg->height = 35;
    	binimg->data = (unsigned char*) malloc(binimg->width*binimg->height*sizeof(char));

    	for(i = 0; i < binimg->width*binimg->height; i++){
    		binimg->data[i] = (line[0] - '0') * 255;
    		line = line + 2;
    	}
    	
    	binimg->val = line[0] - '0';

    	binimg->next = head;
    	head = binimg;

    	n_img++;

    	line = NULL;
    }

	fclose(fp);

	base = head;

	//A base de classificação consiste em 50% da base total
	for(i = 0; i < n_img/2; i++)
		base = base->next;

	printf("Números carregados, extraindo atributos das imagens...\n");

	extract_attr_signature(head, n_img, head->height, head->width);

	printf("Extração concluída, executando treinamento dos classificadores de padrão...\n");

	cv::Ptr<cv::ml::SVM> svm = svm_training(head, n_img/2, head->height, head->width); //Treinando somente com 50% da base

	printf("Treinamento concluído, executando testes...\n");

	classify(svm, base, n_img/2, head->height, head->width); // Classificando usando os outros 50%

	printf("\nTestes concluídos. O resultado foi de %d acertos e %d erros\n", acertos, erros);
	printf("Acurácia");
	printf("Matriz de confusão");
}

void extract_attr_signature(Tbinimg *head, int n_img, int imgx, int imgy){
	int s;

	for(s = 0; s < n_img; s++){
		int px, py, i, j, z;
		int count = 0;
		int oldcount = 0;

		//Algoritmo seguidor de fronteira de Moore
		for(px = 0; px < imgx; px++)
			for(py = 0; py < imgy; py++)
				if(head->data[imgy*py + px] > 0)
					break;
			
		count++;
		i = py;
		j = px;

		while(oldcount != count && (i != py || j != px)){
			oldcount = count;

			if(head->data[imgy*(i-1) + j] > 0){
				i--; count++; continue;
			}
			if(head->data[imgy*(i-1) + j+1] > 0){
				i--; j++; count++; continue;
			}
			if(head->data[imgy*i + j+1] > 0){
				j++; count++; continue;
			}
			if(head->data[imgy*(i+1) + j + 1] > 0){
				i++; j++; count++; continue;
			}
			if(head->data[imgy*(i+1) + j] > 0){
				i++; count++; continue;
			}
			if(head->data[imgy*(i+1) + j - 1] > 0){
				i++; j--; count++; continue;
			}
			if(head->data[imgy*i + j - 1] > 0){
				j--; count++; continue;
			}
			if(head->data[imgy*(i-1) + j - 1] > 0){
				i--; j--; count++; continue;
			}
		}

		head->borders = (TPoint*) malloc(count * sizeof(TPoint));
		head->borders_size = count;

		i = py;
		j = px;

		for(z = 0; z < count; z++){
			if(head->data[imgy*(i-1) + j] > 0){
				head->borders[z].x = j;
				head->borders[z].y = i-1;
				continue;
			}
			if(head->data[imgy*(i-1) + j+1] > 0){
				head->borders[z].x = j+1;
				head->borders[z].y = i-1;
				continue;
			}
			if(head->data[imgy*i + j+1] > 0){
				head->borders[z].x = j+1;
				head->borders[z].y = i;
				continue;
			}
			if(head->data[imgy*(i+1) + j + 1] > 0){
				head->borders[z].x = j+1;
				head->borders[z].y = i+1;
				continue;
			}
			if(head->data[imgy*(i+1) + j] > 0){
				head->borders[z].x = j;
				head->borders[z].y = i+1;
				continue;
			}
			if(head->data[imgy*(i+1) + j - 1] > 0){
				head->borders[z].x = j-1;
				head->borders[z].y = i+1;
				continue;
			}
			if(head->data[imgy*i + j - 1] > 0){
				head->borders[z].x = j-1;
				head->borders[z].y = i;
				continue;
			}
			if(head->data[imgy*(i-1) + j - 1] > 0){
				head->borders[z].x = j-1;
				head->borders[z].y = i-1;
				continue;
			}
		}

		//Iniciando algoritmo de obtenção de assinatura
		head->signature = (unsigned int*) malloc(count*sizeof(int));

		for(z = 0; z < count; z++){
			head->signature[z] = (unsigned int)trunc(sqrt(pow(head->borders[z].x, 2) + pow(head->borders[z].y, 2)));
		}

		head = head->next;
	}
}	

void classify(cv::Ptr<cv::ml::SVM> svm, Tbinimg *head, int n_img, int imgx, int imgy){
	cv::Mat query; // input, 1channel, 1 row (apply reshape(1,1) if nessecary)
	cv::Mat res;   // output
	svm->predict(query, res);
}

cv::Ptr<cv::ml::SVM> svm_training(Tbinimg *head, int n_img, int imgx, int imgy){
	unsigned char labelsarray[n_img];
	//unsigned char *dataarray = (unsigned char*) malloc(n_img*imgx*imgy);
	cv::Mat trainData, labels;
	Tbinimg *temp = head;
	int i;

	for(i = 0; i < n_img; i++){
		labelsarray[i] = head->val;
		//memcpy(dataarray + (i*imgx*imgy), head->data, imgx*imgy);
		temp = temp->next;
	}

	trainData = cv::Mat(head->borders_size, 1, CV_16UC1, head->signature);
	labels = cv::Mat(n_img, 1, CV_8UC1, labelsarray);

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::POLY);
	svm->setGamma(3);

	//cv::Ptr<cv::ml::TrainData> tData = cv::ml::TrainData::create(trainData, cv::ml::SampleTypes::ROW_SAMPLE, labels);

	svm->train(trainData, cv::ml::ROW_SAMPLE, labels);

	//free(dataarray);
}

void menu_reggrowth(char *filename){
	CvMat *img;
	int seed_pos_x, seed_pos_y;
	unsigned char *newimgdata;
	
	if(img = matFromFile(filename)){
		clean_stdin();
		printf("Dimensions: %d x %d \n", img->width, img->height);
		printf("Informe um seed (x,y)\n");
		scanf("%d,%d", &seed_pos_x, &seed_pos_y);
		clean_stdin();

		newimgdata = reggrowth(img->data.ptr, img->width, img->height, seed_pos_x, seed_pos_y);

		CvMat resimg = cvMat(img->height, img->width, CV_8UC1, newimgdata);

		printf("A imagem ficará assim\n");

		cvShowImage( "mainWin", &resimg); 
		cvWaitKey(20);

		confirm_save(filename, &resimg);

		free(newimgdata);
	}
}

void _conv(char *filename, CvMat* img, double **mask, int mask_size_x, int mask_size_y){
	int input_size_x, input_size_y; 
	unsigned char *convolutedData;

	input_size_x = img->width;
	input_size_y = img->height;

	convolutedData = convolution(img->data.ptr, mask, img->width, img->height, mask_size_x, mask_size_y);

	printf("Convolution Complete!\n");

	CvMat resimg = cvMat(input_size_y, input_size_x, CV_8UC1, convolutedData);

	printf("A imagem ficará assim\n");

	cvShowImage( "mainWin", &resimg); 
	cvWaitKey(20);

	if(ask_repeatop()){
		_conv(filename, &resimg, mask, mask_size_x, mask_size_y);
	}
	else{
		confirm_save(filename, &resimg);
	}

	free(convolutedData);
}

//MENU 
void menu_conv(char *filename){
	int mask_size_x, mask_size_y; 
	double **mask;
	
	int params[3] = {CV_IMWRITE_JPEG_QUALITY, 95, 0};

	if(strcmp(filename, "")){
		CvMat* img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

		printf("Dimensions: %d x %d \n", img->width, img->height);

		mask = ask_mask(&mask_size_x, &mask_size_y);	

		_conv(filename, img, mask, mask_size_x, mask_size_y);

		cvReleaseMat(&img);
		free2dArr((void**)mask, mask_size_x);
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
	}	
}

void menu_histogram(char *filename){
	int it;
	char c;
	
	if(strcmp(filename, "")){				
		CvMat* img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

		printf("Dimensions: %d x %d \n", img->width, img->height);

		unsigned int *histogr = histogram(img->data.ptr, img->width, img->height);

		for(it = 0; it < 256; it++){
			printf("%d: %d \n", it, histogr[it]);
		}

		show_histogram(histogr);

		free(histogr);
		cvReleaseMat(&img);
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
	}
}

void menu_equalize(char *filename){
	if(strcmp(filename, "")){
		CvMat* img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

		printf("Dimensions: %d x %d \n", img->width, img->height);

		unsigned int *histogr = histogram(img->data.ptr, img->width, img->height);

		equalize_htgr(histogr, img->data.ptr, img->width, img->height);

		clean_stdin();

		printf("A imagem ficará assim\n");

		cvShowImage( "mainWin", img ); 
		cvWaitKey(100);

		confirm_save(filename, img);

		cvReleaseMat(&img);
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
	}
}

void menu_limiarize(char *filename){
	char opcao;
	unsigned char *limiars;
	unsigned char threshold;
	int window_size_x, window_size_y;
	int pivot_sz;
	unsigned char *newimgdata = NULL;

	if(strcmp(filename, "")){
		CvMat* img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

		printf("Dimensions: %d x %d \n", img->width, img->height);

		printf("\n\t\t\t Selecione o tipo de limiariarização \n\n");
		printf("\t\t0 - Global\n");
		printf("\t\t1 - Local\n");

		clean_stdin();

		printf("Opção: ");
		scanf("%c", &opcao);

		clean_stdin();

		if(opcao == '0'){
			if(!ask_limiars(&limiars, &pivot_sz)){
				newimgdata = limiarize(img->data.ptr, limiars, pivot_sz, img->width, img->height);
			}
		}
		else{
			ask_locallim_data(&threshold, &window_size_x, &window_size_y);
			newimgdata = local_limiarize(img->data.ptr, threshold, img->width, img->height, window_size_x, window_size_y);
		}

		if(newimgdata){
			CvMat resimg = cvMat(img->height, img->width, CV_8UC1, newimgdata);

			printf("A imagem ficará assim\n");

			cvShowImage( "mainWin", &resimg); 
			cvWaitKey(100);

			confirm_save(filename, &resimg);

			free(newimgdata);
		}

		cvReleaseMat(&img);
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
	}
}

void menu_median(char *filename){
	if(strcmp(filename, "")){
		CvMat* img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);
		unsigned char* medianData;

		printf("Dimensions: %d x %d \n", img->width, img->height);

		medianData = median3x3(img->data.ptr, img->width, img->height);

		CvMat resimg = cvMat(img->height, img->width, CV_8UC1, medianData);

		printf("A imagem ficará assim\n");

		cvShowImage( "mainWin", &resimg); 
		cvWaitKey(20);

		confirm_save(filename, &resimg);

		free(medianData);
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
	}
}

double **ask_mask(int *n1, int *n2){
	int i, j;
	printf("Digite o tamanho da máscara (n1, n2)\n");
	scanf("%d, %d", n1, n2);
	double **mask = (double**) create2dArr(*n1, *n2, sizeof(double));

	for(i = 0; i < *n1; i++)
		for(j = 0; j < *n2; j++){
			double tmp;
			clean_stdin();
			printf("Digite o valor (%d, %d):\n", i, j);
			scanf("%lf", &tmp);
			mask[i][j] = tmp;
		}

	return mask;
}


void ask_locallim_data(unsigned char* threshold, int* window_size_x, int* window_size_y){
	printf("Informe o tamanho da janela (N1xN2):\n");
	scanf("%dx%d", window_size_x, window_size_y);

	clean_stdin();

	printf("Informe o limiar:\n");
	scanf("%c", threshold);

	clean_stdin();
}

char ask_limiars(unsigned char** limiars, int* pivot_sz){
	unsigned char* pivots;
	int i, psz;
	int user_input;
	int result = 0;

	printf("Informe a quantidade de limiares:\n");
	scanf("%d", &psz);

	pivots = (unsigned char*) malloc(psz);

	printf("Informe os limiares em ordem crescente\n");

	for(i = 0; i < psz && !result; i++){
		clean_stdin();
		
		printf("Informe o limiar %d:\n", i+1);
		scanf("%d", &user_input);

		pivots[i] = (unsigned char)user_input;

		if(i > 0 && pivots[i] <= pivots[i-1]){
			result = 1;
			printf("O valor %d é menor ou igual ao limiar anterior: %d\n", pivots[i], pivots[i-1]);
		}
	}

	if(!result){
		*limiars = pivots;
		*pivot_sz = psz;
	}
	else{
		free(pivots);
	}	
	
	return result;	
}

void show_and_trysave(char *filename, int height, int width, unsigned char* newimgdata){
	CvMat resimg = cvMat(height, width, CV_8UC1, newimgdata);

	printf("A imagem ficará assim\n");

	cvShowImage( "mainWin", &resimg); 
	cvWaitKey(20);

	confirm_save(filename, &resimg);
}

void confirm_save(char* filename, CvMat *img){
	char ynanswer[1];

	printf("Deseja salvá-la? (s/n)\n");
	scanf("%s", ynanswer);

	if(ynanswer[0] == 's' || ynanswer[0] == 'S'){
		cvSaveImage("out.jpg", img);
		strcpy(filename, "out.jpg");

		printf("\nImagem salva com sucesso!\n\n");
	}
	else{
		printf("Abortando...\n");

		CvMat* bkp = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

		cvShowImage( "mainWin", bkp); 
		cvWaitKey(100);

		cvReleaseMat(&bkp);
	}
}

CvMat *matFromFile(char *filename){
	CvMat *img;

	if(strcmp(filename, "")){
		img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);
		
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
		img = NULL;
	}

	return img;
}

int ask_repeatop(){
	char ynanswer[1];

	printf("Aplicar novamente? (s/n)\n");
	scanf("%s", ynanswer);

	return ynanswer[0] == 's' || ynanswer[0] == 'S';
}

FILE *ask_inputfile(char **filepath){
	char filename[150];
	FILE *loadedFile = NULL;

	printf("Digite o caminho do arquivo\n");
	scanf("%s", filename);
	loadedFile = fopen(filename, "rb");
	if(!loadedFile){
		strcpy(filename, "");
		printf("Não foi possível carregar a imagem!\n");
	}

	strcpy(*filepath, filename);

	return loadedFile;
}

FILE *ask_outputfile(){
	char filename[150];
	FILE *loadedFile = NULL;

	printf("Digite o caminho do arquivo de saída\n");
	scanf("%s", filename);
	loadedFile = fopen(filename, "w+");
	if(!loadedFile){
		strcpy(filename, "");
		printf("Não foi possível criar a imagem!\n");
	}

	return loadedFile;
}

//Utils
void clean_stdin()
{
    int c;
    do {
        c = getchar();
    } while (c != '\n' && c != EOF);
}

unsigned long file_size(FILE *fp){
	int size = 0;
	unsigned char buffer[1024];
	
	while((size+=fread(buffer, 1, 1024, fp)) == 1024);

	rewind(fp);

	return size;
}

void **create2dArr(int x, int y, size_t sz){
	void **arr = (void**) malloc(x * sizeof(void*));
	int i = 0;

	for(i = 0; i < x; i++){
		arr[i] = (void*) malloc(y*sz);
	}

	return arr;
}

void free2dArr(void **arr, int size){
	int i = 0;

	for(; i < size; i++){
		free(arr[i]);
	}

	free(arr);
}

int min(int a, int b){
	return a > b ? b : a;
}

int max(int a, int b){
	return a < b ? b : a;
}

TStack *new_stack(){
	TStack *newstack = (TStack*) malloc(sizeof(TStack));

	newstack->value = (int*) malloc(sizeof(int));
	newstack->value[0] = 0;
	newstack->next = NULL;

	return newstack;
}

void free_stack(TStack *stack){
	TStack *holder;

	do{
		holder = stack;
		stack = stack->next;
		free(holder->value);
		free(holder);
	}while(stack);
}

int stack_size(TStack *stack){
	return stack->value[0];
}

void push(TStack *stack, int va_length, ...){
	TStack *headnode = new_stack();
	int *value = (int*) malloc(sizeof(int)*va_length);
	int i;
	va_list args;

	va_start(args, va_length);

	for(i = 0; i < va_length; i++){
		value[i] = va_arg(args, int);
	}

	headnode->value = value;
	headnode->next = stack->next;

	stack->value[0]++;
	stack->next = headnode;

	va_end(args);
}

int *pop(TStack *stack){
	TStack *oldnode = stack->next;
	int *value = oldnode->value;
	
	stack->value[0]--;
	stack->next = oldnode->next;
	free(oldnode);

	return value;
}

void showImg(const char* id, CvMat *img){
	cvNamedWindow( id, CV_WINDOW_AUTOSIZE ); // Create a window for display.
	cvMoveWindow(id, 100, 100);
	cvShowImage( id, img );  // Show our image inside it.  

	printf("Pressione ENTER para sair...\n");

	while(cvWaitKey(0) != 13);

	cvWaitKey(100);

	cvDestroyWindow(id);
}

void show_histogram(unsigned int *histogram){
	unsigned char *histogr;
	int i;
	CvScalar blackcolo;
	CvScalar whitecolo;

	histogr = normalize(histogram, 256);

	CvMat *mat = cvCreateMat(256, 256, CV_8UC1);

	blackcolo.val[0] = blackcolo.val[1] = blackcolo.val[2] = blackcolo.val[3] = 0;
	whitecolo.val[0] = whitecolo.val[1] = whitecolo.val[2] = whitecolo.val[3] = 255;

	cvRectangle(mat, cvPoint(0, 0), cvPoint(256, 256), whitecolo, CV_FILLED, 8, 0);

	for(i = 0; i < 256; i++)
		cvLine(mat, cvPoint(i, 255-histogr[i]), cvPoint(i, 255), blackcolo, 1, 8, 0);

	cvShowImage( "mainWin", mat); 
	cvWaitKey(100);
}

//**********************************************************BUSINESS********************************************************************************

//Faz a convolução de mask em matrix
unsigned char *convolution(const unsigned char *matrix, double **mask, int matrix_size_x, int matrix_size_y, int mask_size_x, int mask_size_y){
	unsigned char *result;
	int *rawResult = (int*) malloc(matrix_size_y*matrix_size_x*sizeof(int));
	int z, w, i, j;
	int halfsz_x = trunc(mask_size_x / 2);
	int halfsz_y = trunc(mask_size_y / 2);

	for(z = 0; z < matrix_size_y; z++)
		for(w = 0; w < matrix_size_x; w++){
			int xoffset = 0, yoffset = 0, exoffset = 0, eyoffset = 0;

			rawResult[z*matrix_size_x + w] = 0;

			if(z - halfsz_y < 0){
				yoffset = halfsz_y -z;
			}
			if(z + halfsz_y > matrix_size_y){
				eyoffset = (z + halfsz_y) - matrix_size_y;
			}
			if(w - halfsz_x < 0){
				xoffset = halfsz_x -w; 
			}
			if(w + halfsz_x > matrix_size_x){
				exoffset = (w + halfsz_x) - matrix_size_x ;
			}

			for(j = yoffset; j < mask_size_y - eyoffset; j++)
				for(i = xoffset; i < mask_size_x - exoffset; i++){
					rawResult[z*matrix_size_x + w] += round(matrix[(z + j - halfsz_y)*matrix_size_x + (w + i - halfsz_x)] * mask[j][i]);
				}

			rawResult[z*matrix_size_x + w] = abs(rawResult[z*matrix_size_x + w]);
		}
	
	result = normalize((unsigned int*)rawResult, matrix_size_x*matrix_size_y);

	free(rawResult);

	return result;
}

//Transforma uma faixa de valores em outra (no caso, transforma os valores no vetor data em valores entre 0 e 255)
unsigned char *normalize(const unsigned int *data, int data_size){
	unsigned char *result = (unsigned char*) malloc(data_size*sizeof(unsigned char));
	unsigned int highestValue = 0;
	unsigned int lowestValue = pow(2, 32) - 1;
	int i;

	for(i = 0; i < data_size; i++){
		if(highestValue < data[i]){
			highestValue = data[i];
		}
		if(lowestValue > data[i]){
			lowestValue = data[i];
		}
	}

	for(i = 0; i < data_size; i++){
		//round((I - min) * ((newMax - newMin)/(max - min)) + newMin)
		result[i] = round((data[i] - lowestValue)*255/(highestValue - lowestValue));
	}

	return result;
}

unsigned int *histogram(const unsigned char *matrix, int matrix_size_x, int matrix_size_y){
	unsigned int* histogr = (unsigned int*) malloc(256*sizeof(unsigned int));
	int z, w;

	memset(histogr, 0, 256*sizeof(unsigned int));

	for(z = 0; z < matrix_size_y; z++)
		for(w = 0; w < matrix_size_x; w++)
			histogr[matrix[z*matrix_size_x + w]]++;
					
	return histogr;
}

void equalize_htgr(unsigned int *histogram, unsigned char *matrix, int matrix_size_x, int matrix_size_y){
	double *pmf = (double*) malloc(256*sizeof(double));
	double totalpx = 0.0D;
	int i, j;

	for(i = 0; i < 256; i++)
		totalpx += histogram[i];
	for(i = 0; i < 256; i++)
		pmf[i] = histogram[i]/totalpx;
	for(i = 1; i < 256; i++)
		pmf[i] += pmf[i-1];
	for(i = 1; i < 256; i++)
		pmf[i] = 255*pmf[i];
	for(i = 255; i >= 0; i--)
		if(i < pmf[i]){
			histogram[(int)pmf[i]] = histogram[i];
		}

	for(i = 0; i < matrix_size_y; i++){
		for(j = 0; j < matrix_size_x; j++)
			matrix[i*matrix_size_x + j] = (unsigned char) pmf[matrix[i*matrix_size_x + j]];

		////MAGIC!
		// cvShowImage( "mainWin", &cvMat(matrix_size_y, matrix_size_x, CV_8UC1, matrix) ); 
		// cvWaitKey(20);
	}
}

char chkpivot(unsigned char value, unsigned char pivot){
	return abs(pivot - value) < 5;
}

unsigned char *reggrowth(unsigned char* matrix, int matrix_size_x, int matrix_size_y, int seed_pos_x, int seed_pos_y){
	TStack *stack = new_stack();
	int *point;
	int pox, poy;
	unsigned char pivot = matrix[seed_pos_x*matrix_size_x + seed_pos_y];
	unsigned char *result = (unsigned char*) malloc(matrix_size_x*matrix_size_y*sizeof(unsigned char));

	memcpy(result, matrix, matrix_size_x*matrix_size_y);

	push(stack, 2, seed_pos_x, seed_pos_y);

	while(stack_size(stack)){
		point = pop(stack);
		pox = point[0];
		poy = point[1];

		free(point); 

		if(chkpivot(result[poy*matrix_size_x + pox], pivot)){
			result[poy*matrix_size_x + pox] = 0x00;

			if(pox+1 < matrix_size_x)
				if(chkpivot(result[poy*matrix_size_x + pox + 1], pivot))
					push(stack, 2, pox + 1, poy);

			if(pox-1 >= 0)
				if(chkpivot(result[poy*matrix_size_x + pox - 1], pivot))
					push(stack, 2, pox - 1, poy);

			if(pox+1 < matrix_size_x && poy+1 < matrix_size_y)
				if(chkpivot(result[(poy+1)*matrix_size_x + pox + 1], pivot))
					push(stack, 2, pox + 1, poy + 1);

			if(pox-1 >= 0 && poy+1 < matrix_size_y)
				if(chkpivot(result[(poy+1)*matrix_size_x + pox - 1], pivot))
					push(stack, 2, pox - 1, poy + 1);

			if(pox+1 < matrix_size_x && poy-1 >= 0)
				if(chkpivot(result[(poy-1)*matrix_size_x + pox + 1], pivot))
					push(stack, 2, pox + 1, poy - 1);

			if(pox-1 >= 0 && poy-1 >= 0)
				if(chkpivot(result[(poy-1)*matrix_size_x + pox - 1], pivot))
					push(stack, 2, pox - 1, poy - 1);

			if(poy+1 < matrix_size_y)
				if(chkpivot(result[(poy+1)*matrix_size_x + pox], pivot))
					push(stack, 2, pox, poy+1);

			if(poy-1 >= 0)
				if(chkpivot(result[(poy-1)*matrix_size_x + pox], pivot))
					push(stack, 2, pox, poy-1);
		}
	}

	free_stack(stack);

	return result;
}

unsigned char *limiarize(unsigned char *matrix, unsigned char *pivot, int pivot_sz, int matrix_size_x, int matrix_size_y){
	int i, j, z, nomatch = 1;
	unsigned char *result = (unsigned char*) malloc(matrix_size_x*matrix_size_y*sizeof(unsigned char));

	for(i = 0; i < matrix_size_y; i++)
		for(j = 0; j < matrix_size_x; j++)
			for(z = 0, nomatch = 1; z < pivot_sz && nomatch; z++) 
				if(z == 0 && matrix[i*matrix_size_x + j] < pivot[z]){
					result[i*matrix_size_x + j] = 0;
					nomatch = 0;
				}
				else if(matrix[i*matrix_size_x + j] < pivot[z]) {
					result[i*matrix_size_x + j] = (pivot[z-1] + pivot[z])/2;
					nomatch = 0;
				}
				else if((z+1) == pivot_sz && matrix[i*matrix_size_x + j] > pivot[z]){
					result[i*matrix_size_x + j] = 255;
					nomatch = 0;
				}

	return result;
}

unsigned char *local_limiarize(unsigned char *matrix, unsigned char threshold, int matrix_size_x, int matrix_size_y, int window_size_x, int window_size_y){
	int i, j, w, z;
	int halfsz_x = window_size_x / 2;
	int halfsz_y = window_size_y / 2;
	int xoffset, yoffset, exoffset, eyoffset;
	int maxgray, mingray, midgray;
	unsigned char *result = (unsigned char*) malloc(matrix_size_x*matrix_size_y*sizeof(unsigned char));

	for(z = 0; z < matrix_size_y; z++)
		for(w = 0; w < matrix_size_x; w++){
			xoffset = 0; yoffset = 0; exoffset = 0; eyoffset = 0;
			maxgray = 0x00; mingray = 0xFF; midgray = 0x00;

			if(z - halfsz_y < 0){
				yoffset = halfsz_y -z;
			}
			if(z + halfsz_y > matrix_size_y){
				eyoffset = (z + halfsz_y) - matrix_size_y;
			}
			if(w - halfsz_x < 0){
				xoffset = halfsz_x -w; 
			}
			if(w + halfsz_x > matrix_size_x){
				exoffset = (w + halfsz_x) - matrix_size_x ;
			}

			for(j = yoffset; j < window_size_y - eyoffset; j++)
				for(i = xoffset; i < window_size_x - exoffset; i++){
					int val = matrix[(z + j - halfsz_y)*matrix_size_x + (w + i - halfsz_x)];

					if(maxgray < val)
						maxgray = val;
					if(mingray > val)
						mingray = val;
				}

			midgray = (maxgray + mingray) / 2;
			result[z*matrix_size_x + w] = midgray >= threshold ? 0xFF : 0x00;
		}

	return result;
}

unsigned char *median3x3(const unsigned char *matrix1, int matrix_size_x, int matrix_size_y){
	unsigned char *result = (unsigned char*) malloc(matrix_size_x*matrix_size_y);
	int neighborhood[8];
	int i, j, z, w, aux;

	for(i = 0; i < matrix_size_y; i++)
		for(j = 0; j < matrix_size_x; j++){
			neighborhood[0] = neighborhood[1] = neighborhood[2] = neighborhood[3] =
			neighborhood[4] = neighborhood[5] = neighborhood[6] = neighborhood[7] = -1;

			if(i - 1 >= 0 && j - 1 >= 0)
				neighborhood[0] = matrix1[(i-1)*matrix_size_x + j-1];
			if(i - 1 >= 0)
				neighborhood[1] = matrix1[(i-1)*matrix_size_x + j];
			if(i - 1 >= 0 && j + 1 < matrix_size_x)
				neighborhood[2] = matrix1[(i-1)*matrix_size_x + j + 1];
			if(j - 1 >= 0)
				neighborhood[3] = matrix1[i*matrix_size_x + j - 1];
			if(j + 1 < matrix_size_x)
				neighborhood[4] = matrix1[i*matrix_size_x + j + 1];
			if(i + 1 < matrix_size_y && j - 1 >= 0)
				neighborhood[5] = matrix1[(i+1)*matrix_size_x + j - 1];
			if(i + 1 < matrix_size_y)
				neighborhood[6] = matrix1[(i+1)*matrix_size_x + j];
			if(i + 1 < matrix_size_y && j + 1 < matrix_size_x)
				neighborhood[7] = matrix1[(i+1)*matrix_size_x + j + 1];

			for(z = 0; z < 8; z++)
				for(w = z+1; w < 8; w++){
					if(neighborhood[z] > neighborhood[w]){
						aux = neighborhood[z];
						neighborhood[z] = neighborhood[w];
						neighborhood[w] = aux;
					}
				}

			for(z = 0; neighborhood[z] == -1; z++);

			result[i*matrix_size_x + j] = (unsigned char) ((8+z) % 2 == 0) ? ((neighborhood[(8+z)/2] + neighborhood[1 + (8+z)/2]) / 2) : (neighborhood[1 + (8+z)/2]) ; 
		}

	return result;
}

unsigned char *hough_transform(const unsigned char *matrix, int matrix_size_x, int matrix_size_y){
	double hough_h = sqrt(2.0) * max(matrix_size_x, matrix_size_y) / 2.0;
	unsigned int *rawResult = (unsigned int*) malloc(((int)hough_h)*180*2*sizeof(int));
	unsigned char *result;
	int i, j, angle;
	int centerx = matrix_size_x/2;
	int centery = matrix_size_y/2;
	double r;

	memset(rawResult, 0, ((int)hough_h)*180*2*sizeof(int));

	for(i = 0; i < matrix_size_y; i++)
		for(j = 0; j < matrix_size_x; j++)
			if(matrix[ (i*matrix_size_x) + j] >= 250)
				for(angle = 0; angle < 180; angle++){
					r = (j-centerx) * cos(angle * M_PI / 180) + (i-centery) * sin(angle * M_PI / 180);
					rawResult[(int)round(angle*hough_h*2) + ((int)round(r + hough_h))]++; 
				}


	result = normalize(rawResult, ((int)hough_h)*180*2);

	free(rawResult);

	return result;
}

unsigned char *sum_images(const unsigned char *matrix1, const unsigned char *matrix2, int min_x_m1_m2, int min_y_m1_m2){
	unsigned int *rawResult = (unsigned int*) malloc(min_x_m1_m2*min_y_m1_m2*sizeof(int));
	unsigned char *result;
	int i, j;

	for(i = 0; i < min_y_m1_m2; i++)
		for(j = 0; j < min_x_m1_m2; j++)
			rawResult[i*min_x_m1_m2 + j] = ((unsigned int)matrix1[i*min_x_m1_m2 + j]) + matrix2[i*min_x_m1_m2 + j];

	result = normalize(rawResult, min_x_m1_m2*min_y_m1_m2);

	free(rawResult);

	return result;
}

unsigned char *radquadsum_images(const unsigned char *matrix1, const unsigned char *matrix2, int min_x_m1_m2, int min_y_m1_m2){
	unsigned int *rawResult = (unsigned int*) malloc(min_x_m1_m2*min_y_m1_m2*sizeof(int));
	unsigned char *result;
	int i, j;

	for(i = 0; i < min_y_m1_m2; i++)
		for(j = 0; j < min_x_m1_m2; j++)
			rawResult[i*min_x_m1_m2 + j] = sqrt(((unsigned int)matrix1[i*min_x_m1_m2 + j])*matrix1[i*min_x_m1_m2 + j] + ((unsigned int)matrix2[i*min_x_m1_m2 + j])*matrix2[i*min_x_m1_m2 + j]);

	result = normalize(rawResult, min_x_m1_m2*min_y_m1_m2);

	free(rawResult);

	return result;
}




// //Business functions
// void fconvolution(FILE *input, FILE *output, unsigned char **mask, int input_size_x, int input_size_y, int input_size, int mask_size_x, int mask_size_y){
// 	int w = 0, z = 0, x, y, j;	
// 	unsigned char image_match[mask_size_x][mask_size_y];
// 	int buffer_size =  input_size_x * (mask_size_y /2 + 1);  //128 * mask_size_x * mask_size_y;
// 	int readbuffer[buffer_size];
// 	int bytesread = 0;
// 	int writebuffer[buffer_size];
// 	int writeidx = 0;

	
// 	printf("Tamanho da imagem: %d x %d\n", input_size_x, input_size_y);
// 	printf("Tamanho da máscara: %d x %d\n", mask_size_x, mask_size_y);

// 	for(; z < input_size_y; z++){
// 		fseek(input, z*input_size_x, SEEK_CUR);
// 		bytesread += fread(readbuffer, 1, buffer_size, input);

// 		for(; w < input_size_x; w++){
// 			int xoffset = 0, yoffset = 0, exoffset = 0, eyoffset = 0;

// 			memset(image_match, 0, mask_size_x*mask_size_y);

// 			if(z+1 - mask_size_y < 0){
// 				yoffset = mask_size_y -z-1;
// 			}
// 			if(z+1 + mask_size_y > input_size_y){
// 				eyoffset = (z+1 + mask_size_y) - input_size_y;
// 			}
// 			if(w+1 - mask_size_x < 0){
// 				xoffset = mask_size_x -w-1; 
// 			}
// 			if(w+1 + mask_size_x > input_size_x){
// 				exoffset = (w+1 + mask_size_x) - input_size_x ;
// 			}

// 			for(j = yoffset; j < mask_size_y - eyoffset; j++){
// 				int yread = (j - 2);
// 				int xread = w - xoffset - mask_size_x / 2;


// 				//TODO talvez precise eliminar esse for...
// 				image_match[yread][j] = readbuffer[xread + yread*input_size_x];


// 				//fseek(input, yread + xread, SEEK_SET);

// 				//fread(image_match + (j*mask_size_y + xoffset), 1, mask_size_x - xoffset, input);
// 			}

// 			for(y = 0; y < mask_size_y; y++)
// 				for(x = 0; x < mask_size_x; x++){
// 					writebuffer[writeidx] += image_match[x][y] * mask[x][y];
// 				}

// 			writeidx++;

// 			if(writeidx >= buffer_size){
// 				writeidx = 0;
// 				fwrite(writebuffer, 4, buffer_size/4, output);
// 			}
// 		}
// 	}
	
// 	if(writeidx){
// 		fwrite(writebuffer, 1, writeidx, output);
// 	}
// }




