#include "pathtracer.h"
#include "camera.h"
#include "scene.h"
#include "bvh.h"
#include "imageio.h"
#include "parsescene.h"
#include <gl\glew.h>
#include <gl\glut.h>
#include <time.h>
#include <cuda_gl_interop.h>
//#include <OpenImageDenoise\oidn.h>

const char* title = "FuHong's GPU Pathtracer";

GlobalConfig config;
unsigned iteration = 0;
bool vision_bvh = false;
bool reset_acc_image = false;
clock_t start = 0, last = 0;
float3* image, *dev_ptr;
Scene scene;
GLuint buffer;
cudaGraphicsResource* resource = NULL;

////intel open image denoise
////https://openimagedenoise.github.io/documentation.html
//void Denoiser(float3* colorPtr, float3* outputPtr){
//	OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
//	oidnCommitDevice(device);
//
//	// Create a denoising filter
//	OIDNFilter filter = oidnNewFilter(device, "RT"); // generic ray tracing filter
//	oidnSetSharedFilterImage(filter, "color", colorPtr,
//		OIDN_FORMAT_FLOAT3, config.width, config.height, 0, 0, 0);
//	//oidnSetSharedFilterImage(filter, "albedo", albedoPtr,
//	//	OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
//	//oidnSetSharedFilterImage(filter, "normal", normalPtr,
//	//	OIDN_FORMAT_FLOAT3, width, height, 0, 0, 0); // optional
//	oidnSetSharedFilterImage(filter, "output", outputPtr,
//		OIDN_FORMAT_FLOAT3, config.width, config.height, 0, 0, 0);
//	oidnSetFilter1b(filter, "hdr", false); // image is HDR
//	oidnCommitFilter(filter);
//
//	// Filter the image
//	oidnExecuteFilter(filter);
//
//	// Check for errors
//	const char* errorMessage;
//	if (oidnGetDeviceError(device, &errorMessage) != OIDN_ERROR_NONE)
//		printf("Error: %s\n", errorMessage);
//
//	// Cleanup
//	oidnReleaseFilter(filter);
//	oidnReleaseDevice(device);
//}

void SaveImage(){
	glReadPixels(0, 0,config.width, config.height, GL_RGB, GL_FLOAT, image);
	char buffer[2048] = { 0 };
	
	vector<float3> output(config.width*config.height);
	//Denoiser(image, &output[0]);
	sprintf(buffer, "../result/%ds iteration %dpx-%dpx.png", iteration, config.width, config.height);
	ImageIO::SavePng(buffer, config.width, config.height, &image[0]);
}

void InitOpengl(int argc, char**argv){
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
	glutInitWindowSize(config.width, config.height);
	glutCreateWindow(title);
	glewInit();

	glViewport(0, 0, config.width, config.height);
	glMatrixMode(GL_PROJECTION);
	gluPerspective(config.camera.fov, float(config.width) / float(config.height), 0.1, 1000);

	glGenBuffers(1, &buffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, buffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, config.width*config.height*sizeof(float3), 0, GL_DYNAMIC_DRAW);

	HANDLE_ERROR(cudaGraphicsGLRegisterBuffer(&resource, buffer, cudaGraphicsMapFlagsNone));
}

void draw_bbox(BBox& bbox){
	float3 min = bbox.fmin;
	float3 max = bbox.fmax;

	glColor3f(0, 0, 1);
	glBegin(GL_LINES);
	glVertex3f(min.x, min.y, min.z);
	glVertex3f(max.x, min.y, min.z);

	glVertex3f(max.x, min.y, min.z);
	glVertex3f(max.x, max.y, min.z);

	glVertex3f(max.x, max.y, min.z);
	glVertex3f(min.x, max.y, min.z);

	glVertex3f(min.x, max.y, min.z);
	glVertex3f(min.x, min.y, min.z);

	glVertex3f(max.x, max.y, max.z);
	glVertex3f(min.x, max.y, max.z);

	glVertex3f(min.x, max.y, max.z);
	glVertex3f(min.x, min.y, max.z);

	glVertex3f(min.x, min.y, max.z);
	glVertex3f(max.x, min.y, max.z);

	glVertex3f(max.x, min.y, max.z);
	glVertex3f(max.x, max.y, max.z);

	glVertex3f(min.x, min.y, min.z);
	glVertex3f(min.x, min.y, max.z);

	glVertex3f(min.x, max.y, min.z);
	glVertex3f(min.x, max.y, max.z);

	glVertex3f(max.x, max.y, max.z);
	glVertex3f(max.x, max.y, min.z);

	glVertex3f(max.x, min.y, min.z);
	glVertex3f(max.x, min.y, max.z);
	glEnd();
}

void visualize_bvh(){
	for (int i = 0; i < scene.bvh.total_nodes; ++i)
		draw_bbox(scene.bvh.linear_root[i].bbox);
}

void start_tracing(){
	size_t size;
	cudaGraphicsMapResources(1, &resource, NULL);
	cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, resource);

	Render(scene, config.width, config.height, scene.camera, iteration, reset_acc_image, dev_ptr);
	reset_acc_image = false;

	//HANDLE_ERROR(cudaMemcpy(image, dev_ptr, width*height*sizeof(float3), cudaMemcpyDeviceToHost));
	cudaGraphicsUnmapResources(1, &resource, NULL);
}

void record_time(){
	clock_t now = clock();

	float delta = (float)(now - last) / CLOCKS_PER_SEC;
	float elapsed = (float)(now - start) / CLOCKS_PER_SEC;

	last = now;
	
	char buffer[128];
	sprintf(buffer, "%s  Seconds:[%.2fs], Fps:[%.2f], Iteration:[%d]", title, elapsed, 1.f / delta, iteration);
	//printf("Seconds:[%.2fs], Fps:[%d], Iteration:[%d]\r\n", elapsed, fps, iteration);
	glutSetWindowTitle(buffer);
}

void update_camera(){
	glMatrixMode(GL_MODELVIEW);
	GLfloat matrix[16];
	matrix[0] = scene.camera->u.x; matrix[1] = scene.camera->v.x; matrix[2] = scene.camera->w.x; matrix[3] = 0.f;
	matrix[4] = scene.camera->u.y; matrix[5] = scene.camera->v.y; matrix[6] = scene.camera->w.y; matrix[7] = 0.f;
	matrix[8] = scene.camera->u.z; matrix[9] = scene.camera->v.z; matrix[10] = scene.camera->w.z; matrix[11] = 0.f;
	matrix[12] = -dot(scene.camera->u, scene.camera->position); matrix[13] = -dot(scene.camera->v, scene.camera->position); matrix[14] = -dot(scene.camera->w, scene.camera->position); matrix[15] = 1.f;
	
	glLoadMatrixf(matrix);
}

static void display(void){
	//increase interation 
	++iteration;

	if (reset_acc_image){
		iteration = 1;
	}
	//start tracing and copy buffer to opengl
	start_tracing();

	//print time
	record_time();

	update_camera();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glDrawPixels(config.width, config.height, GL_RGB, GL_FLOAT, NULL);

	if (vision_bvh)
		visualize_bvh();

	glutSwapBuffers();
	glutPostRedisplay();
}

static void keyboard(unsigned char key, int x, int y){
	float speed = config.camera_move_speed;
	switch (key){
	case 'p':
	case 'P':
		SaveImage();
		break;
	case 'v':
	case 'V':
		vision_bvh = !vision_bvh;
		break;
	case 'w':
	case 'W':
		scene.camera->position -= scene.camera->w*speed;
		reset_acc_image = true;
		break;
	case 's':
	case 'S':
		scene.camera->position += scene.camera->w*speed;
		reset_acc_image = true;
		break;
	case 'a':
	case 'A':
		scene.camera->position -= scene.camera->u*speed;
		reset_acc_image = true;
		break;
	case 'd':
	case 'D':
		scene.camera->position += scene.camera->u*speed;
		reset_acc_image = true;
		break;
	case 'e':
	case 'E':
		scene.camera->position += scene.camera->v*speed;
		reset_acc_image = true;
		break;
	case 'q':
	case 'Q':
		scene.camera->position -= scene.camera->v*speed;
		reset_acc_image = true;
		break;
	case 27:
		EndRender();
		HANDLE_ERROR(cudaFree(dev_ptr));

		exit(0);
	}
}

static void mouse(int button, int state, int x, int y){

}

static void motion(int x, int y){

}

bool InitScene(string file){
	clock_t now = clock();
	if (!LoadScene(file.c_str(), config, scene)){
		fprintf(stderr, "Failed to load scene\n");
		return false;
	}

	Camera cam = config.camera;
	Camera* camera = new Camera(cam.position, cam.u, cam.v, cam.w, make_float2(config.width, config.height), 0.1f, cam.fov, cam.apertureRadius, cam.focalDistance, cam.filmic, cam.medium);
	camera->environment = cam.environment;
	//init light distribution
	scene.Init(camera);

	printf("Load scene using %.3fms\n", float(clock() - now));
	printf("Primitives [%d]\n", scene.bvh.prims.size());

	return true;
}

int main(int argc, char**argv){
	if (argc < 2)
		return 1;

	string f = argv[1];

	//first load scene, important
	if (!InitScene(f)){
		return 1;
	}

	//then init opengl
	InitOpengl(argc, argv);

	start = clock();
	last = start;

	image = new float3[config.width*config.height];
	HANDLE_ERROR(cudaMalloc(&dev_ptr, config.width*config.height*sizeof(float3)));

	BeginRender(scene, config.width, config.height, config.epsilon, config.maxDepth);

	srand(time(NULL));

	glutKeyboardFunc(keyboard);
	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutMainLoop();

	return 0;
}