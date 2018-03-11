#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <opencv/cv.h>
#include <highgui.h>

//Input functions
double **ask_mask(int*, int*);
char ask_limiars(unsigned char**, int*);
FILE *ask_outputfile();
FILE *ask_inputfile(char **filepath);

//Utils
unsigned long file_size(FILE*);
void **create2dArr(int, int, size_t);
void free2dArr(void **arr, int size);
void showImg(const char*, CvMat*);

//Menus
int  menu_loadimage(char *filename, char* bkpfilename);
void menu_showimage(char *filename);
void menu_conv(char *filename);
void menu_histogram(char *filename);
void menu_equalize(char *filename);
void menu_limiarize(char *filename);

//Business Functions
unsigned char *convolution(const unsigned char *matrix, double **mask, int matrix_size_x, int matrix_size_y, int mask_size_x, int mask_size_y);
unsigned char *normalize(const unsigned int *data, int data_size);
unsigned int *histogram(const unsigned char *matrix, int matrix_size_x, int matrix_size_y);
void equalize_htgr(unsigned int *histogram, unsigned char *matrix, int matrix_size_x, int matrix_size_y);
void limiarize(unsigned char *matrix, unsigned char *pivot, int pivot_sz, int matrix_size_x, int matrix_size_y);
void show_histogram(unsigned int *histogram);


//TODO
//Imagem do histograma: implementar função show_histogram
//Pensar em um nome para a função _conv e integrá-la no projeto


//Input Functions
int main(int argsize, char **args){
	int acao;
	char *filename = (char*) malloc(150);
	int it, jt;
	char *bkpfilename = (char*) malloc(150);
	CvMat* img;

	strcpy(filename, "");
	strcpy(bkpfilename, "");

	cvNamedWindow( "mainWin", CV_WINDOW_AUTOSIZE ); // Create a window for display.
	cvMoveWindow("mainWin", 100, 100);

	while(acao != 9){
		fflush(stdin);
		fflush(stdout);

		printf("\n\t\t\t Image toolset \n\n");
		printf("\t\tEscolha uma ação: \n");
		printf("\t\t0 - Carregar imagem LENNA.JPG\n");
		printf("\t\t1 - Carregar uma imagem\n");
		printf("\t\t2 - Exibir Imagem\n");
		printf("\t\t3 - Exibir Histograma\n");
		printf("\t\t4 - Convolução\n");
		printf("\t\t5 - Equalização\n");
		printf("\t\t6 - Limiariarização\n");
		printf("\t\t9 - Sair\n");

		printf("Opção: ");
		scanf("%d", &acao);

		switch(acao){
			case 0 : 
				strcpy(filename, "LENNA.JPG"); 
				strcpy(bkpfilename, filename);

				img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

				cvShowImage( "mainWin", img ); 
				cvWaitKey(100);
			break;

			case 1 : menu_loadimage(filename, bkpfilename); break;
			case 2 : menu_showimage(filename);  			break;
			case 3 : menu_histogram(filename);  			break;
			case 4 : menu_conv(filename); 					break;
			case 5 : menu_equalize(filename) ; 				break;
			case 6 : menu_limiarize(filename); 				break;
		}
	}

	free(filename);
	free(bkpfilename);

	return 0;
} 

int menu_loadimage(char *filename, char* bkpfilename){
	FILE *loadedFile = ask_inputfile(&filename);
	int result = loadedFile != NULL;

	free(loadedFile);
	//strcpy(filename, "LENNA.JPG");

	if(result){
		strcpy(bkpfilename, filename);
		CvMat *img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

		cvShowImage( "mainWin", img ); 
		cvWaitKey(100);
	}
	else{
		printf("Não foi possível carregar o arquivo %s", filename);
		strcpy(filename, bkpfilename);
	}

	return result;
}

void menu_showimage(char *filename){
	if(strcmp(filename, "")){
		CvMat *img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);
		cvShowImage( "mainWin", img ); 	
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
	}
}

void _conv(char *filename, CvMat* img, double **mask, int mask_size_x, int mask_size_y){
	char ynanswer[1];
	int input_size_x, input_size_y; 
	unsigned char *convolutedData;

	input_size_x = img->width;
	input_size_y = img->height;

	convolutedData = convolution(img->data.ptr, mask, img->width, img->height, mask_size_x, mask_size_y);

	printf("Convolution Complete!\n");

	CvMat resimg = cvMat(input_size_y, input_size_x, CV_8UC1, convolutedData);

	printf("A imagem ficará assim\n");

	cvShowImage( "mainWin", &resimg); 
	cvWaitKey(100);

	printf("Aplicar novamente? (s/n)\n");
	scanf("%s", ynanswer);

	if(ynanswer[0] == 's' || ynanswer[0] == 'S'){
		img = &resimg;

		_conv(filename, &resimg, mask, mask_size_x, mask_size_y);
	}
	else{
		printf("Deseja salvá-la? (s/n)\n");
		scanf("%s", ynanswer);

		if(ynanswer[0] == 's' || ynanswer[0] == 'S'){
			cvSaveImage("out.jpg", &resimg);
			strcpy(filename, "out.jpg");
		}
		else{
			printf("Abortando...\n");

			CvMat* bkp = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

			cvShowImage( "mainWin", bkp); 
			cvWaitKey(100);

			cvReleaseMat(&bkp);
		}
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
	char ynanswer[1];

	if(strcmp(filename, "")){
		CvMat* img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

		printf("Dimensions: %d x %d \n", img->width, img->height);

		unsigned int *histogr = histogram(img->data.ptr, img->width, img->height);

		equalize_htgr(histogr, img->data.ptr, img->width, img->height);

		printf("A imagem ficará assim\n");

		cvShowImage( "mainWin", img ); 
		cvWaitKey(100);

		fflush(stdin);
		fflush(stdout);

		printf("Deseja salvá-la? (s/n)\n");
		scanf("%s", ynanswer);

		if(ynanswer[0] == 's' || ynanswer[0] == 'S'){
			cvSaveImage("out.jpg", img);

			strcpy(filename, "out.jpg");

			printf("\nImagem equalizada com sucesso!\n\n");
		}
		else{
			printf("Abortando...\n");

			CvMat* bkp = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

			cvShowImage( "mainWin", bkp); 
			cvWaitKey(100);

			cvReleaseMat(&bkp);
		}

		cvReleaseMat(&img);
	}
	else{
		printf("ERRO: nenhum arquivo carregado!\n");
	}
}

void menu_limiarize(char *filename){
	unsigned char *limiars;
	int pivot_sz;
	char ynanswer[1];

	if(strcmp(filename, "")){
		CvMat* img = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

		printf("Dimensions: %d x %d \n", img->width, img->height);

		if(!ask_limiars(&limiars, &pivot_sz)){
			limiarize(img->data.ptr, limiars, pivot_sz, img->width, img->height);

			printf("A imagem ficará assim\n");

			cvShowImage( "mainWin", img ); 
			cvWaitKey(100);

			printf("Deseja salvá-la? (s/n)\n");
			scanf("%s", ynanswer);

			if(ynanswer[0] == 's' || ynanswer[0] == 'S'){
				cvSaveImage("out.jpg", img);
				strcpy(filename, "out.jpg");

				printf("\nImagem equalizada com sucesso!\n\n");
			}
			else{
				printf("Abortando...\n");

				CvMat* bkp = cvLoadImageM(filename, CV_LOAD_IMAGE_GRAYSCALE);

				cvShowImage( "mainWin", bkp); 
				cvWaitKey(100);

				cvReleaseMat(&bkp);
			}
		}

		cvReleaseMat(&img);
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
			fflush(stdin);
			fflush(stdout);
			printf("Digite o valor (%d, %d):\n", i, j);
			scanf("%lf", &tmp);
			mask[i][j] = tmp;
		}

	return mask;
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
		fflush(stdout);
		fflush(stdin);
		
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

void limiarize(unsigned char *matrix, unsigned char *pivot, int pivot_sz, int matrix_size_x, int matrix_size_y){
	int i, j, z, nomatch = 1;

	for(i = 0; i < matrix_size_y; i++)
		for(j = 0; j < matrix_size_x; j++)
			for(z = 0, nomatch = 1; z < pivot_sz && nomatch; z++) 
				if(z == 0 && matrix[i*matrix_size_x + j] < pivot[z]){
					matrix[i*matrix_size_x + j] = 0;
					nomatch = 0;
				}
				else if(matrix[i*matrix_size_x + j] < pivot[z]) {
					matrix[i*matrix_size_x + j] = pivot[z-1];
					nomatch = 0;
				}
				else if((z+1) == pivot_sz && matrix[i*matrix_size_x + j] > pivot[z]){
					matrix[i*matrix_size_x + j] = 255;
					nomatch = 0;
				}
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




