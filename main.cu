//Using SDL and standard IO
#include <SDL.h>
#include <stdio.h>
#include <math.h>

//Screen dimension constants
const int SCREEN_WIDTH = 1280;
const int SCREEN_HEIGHT = 720;

#define FPI (3.141592653589f)
#define MAX_RAYS 8
#define EPSILON 0.001f
#define FOV (FPI / 3)
#define MOVESPEED 2

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#pragma pack(push,1)
struct color {
	uint8_t b;
	uint8_t g;
	uint8_t r;
	uint8_t a;
};
typedef struct color Color;
#pragma pack(pop)

struct vec3 {
	float x,y,z;
};
typedef struct vec3 Vec3;

struct vec2 {
	float x, y;
};
typedef struct vec2 Vec2;

struct mat3 {
	vec3 rows[3];
};
typedef struct mat3 Mat3;

struct ray {
	Vec3 origin;
	Vec3 direction;
};
typedef struct ray Ray;

enum primtype {
	PRIM_SPHERE,
	PRIM_PLANE
};
typedef enum primtype PrimType;

struct sphere {
	Vec3 centre;
	float radius;
};
typedef struct sphere Sphere;

struct plane {
	Vec3 point;
	Vec3 normal;
	Vec3 up;
	Vec3 right;
};
typedef struct plane Plane;

struct prim {
	PrimType type;
	Color color;
	Color color2;
	float reflectivity;
	union{
		Sphere sphere;
		Plane plane;
	};
};
typedef struct prim Prim;

enum geotype {
	GEO_PRIM,
	GEO_SUB,
	GEO_AABB,
	GEO_PORTAL
};
typedef enum geotype GeoType;

struct geometry{
	GeoType type;
	union{
		struct{
			Vec3 min, max;
			struct geometry* subGeometry;
			int subGeometrySize;
		};
		struct{
			Prim* prim;
			Vec3 dest;
		};
	};
};
typedef struct geometry Geometry;

__device__
Geometry* world;

__device__
int worldSize;

__device__
float screendist, xmult, ymult;

__device__
Vec3 sunDir = {0.57735026919f, 0.57735026919f, 0.57735026919f};

#define dot(a, b) ((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)

#define add(a, b) {(a).x + (b).x, (a).y + (b).y, (a).z + (b).z}

#define sub(a, b) {(a).x - (b).x, (a).y - (b).y, (a).z - (b).z}

#define smul(a, b) {(a).x * (b), (a).y * (b), (a).z * (b)}

#define neg(a) {-(a).x, -(a).y, -(a).z}

__device__ inline
void normalize(Vec3* vec){
	float norm = norm3df(vec->x, vec->y, vec->z);
	vec->x /= norm;
	vec->y /= norm;
	vec->z /= norm;
}

#define blend(j, js, k, ks) {(uint8_t)(((j.b) * (js) + (k.b) * (ks)) / ((js) + (ks))), (uint8_t)(((j.g) * (js) + (k.g) * (ks)) / ((js) + (ks))), (uint8_t)(((j.r) * (js) + (k.r) * (ks)) / ((js) + (ks))), (uint8_t)(((j.a) * (js) + (k.a) * (ks)) / ((js) + (ks)))}

__device__ inline
Vec3 vecMatMul(Mat3 matrix, Vec3 vec){
	return {dot(vec, matrix.rows[0]), dot(vec, matrix.rows[1]), dot(vec, matrix.rows[2])};
}

__device__ inline
Mat3 transpose(Mat3 mat){
	return {
		{
			{mat.rows[0].x, mat.rows[1].x, mat.rows[2].x},
			{mat.rows[0].y, mat.rows[1].y, mat.rows[2].y},
			{mat.rows[0].z, mat.rows[1].z, mat.rows[2].z}
		}
	};
}

__device__ inline
Mat3 matMul(Mat3 a, Mat3 b){
	Mat3 transB = transpose(b);
	return transpose({
		{
			vecMatMul(transB, a.rows[0]),
			vecMatMul(transB, a.rows[1]),
			vecMatMul(transB, a.rows[2])
		}
	});
}

__device__ inline
Mat3 rotatex(float theta){
	return {{
		{1.f, 0.f, 0.f},
		{0.f, cosf(theta), -sinf(theta)},
		{0.f, sinf(theta), cosf(theta)}}
	};
}

__device__ inline
Mat3 rotatey(float theta){
	return {{
		{cosf(theta), 0.f, sinf(theta)},
		{0.f, 1.f, 0.f},
		{-sinf(theta), 0.f, cosf(theta)}}
	};
}

__device__ inline
Mat3 rotatez(float theta){
	return {{
		{cosf(theta), -sinf(theta), 0.f},
		{sinf(theta), cosf(theta), 0.f},
		{0.f, 0.f, 1.f}}
	};
}

__device__ inline
int sphereIntersect(Ray* r, Sphere* s, Vec3* t, Vec3* normal, float* d, Vec2* uv)
{
	Vec3 diff = sub(r->origin, s->centre);
	float uoc = dot(r->direction, diff);
	float w = (uoc * uoc) - (dot(diff, diff) - s->radius * s->radius);
	if(w < 0) return 0;
	float d1 = -uoc + sqrtf(w);
	float d2 = -uoc - sqrtf(w);
	*d = (d1 < d2) ? d1 : d2;
	Vec3 ta = smul(r->direction, *d);
	*t = add(r->origin, ta);
	*normal = sub(*t, s->centre);
	normalize(normal);
	/*if(dot(r->direction, *normal) >= 0)
		*d = INFINITY;*/
	return *d > 0;
}

__device__ inline
int antiSphereIntersect(Ray* r, Sphere* s, Vec3* t, Vec3* normal, float* d, Vec2* uv)
{
	Vec3 diff = sub(r->origin, s->centre);
	float uoc = dot(r->direction, diff);
	float w = (uoc * uoc) - (dot(diff, diff) - s->radius * s->radius);
	if(w < 0) return 0;
	float d1 = -uoc + sqrtf(w);
	float d2 = -uoc - sqrtf(w);
	*d = (d1 > d2) ? d1 : d2;
	Vec3 ta = smul(r->direction, *d);
	*t = add(r->origin, ta);
	*normal = sub(s->centre, *t);
	normalize(normal);
	if(dot(r->direction, *normal) >= 0)
		*d = INFINITY;
	return *d > 0;
}

__device__ inline
int sphereInside(Sphere* s, Vec3* p){
	Vec3 diff = sub(s->centre, *p);
	return dot(diff, diff) < (s->radius * s->radius);
}

__device__ inline
int planeIntersect(Ray* r, Plane* p, Vec3* t, Vec3* normal, float* d, Vec2* uv){
	Vec3 diff = sub(p->point, r->origin);
	*d = dot(diff, p->normal) / dot(r->direction, p->normal);
	Vec3 ta = smul(r->direction, *d);
	*t = add(r->origin, ta);
	*normal = p->normal;
	Vec3 planespace = sub(*t, p->point);
	*uv = {dot(planespace, p->right), dot(planespace, p->up)};
	return (*d > 0) && (dot(r->direction, *normal) < 0);
}

__device__ inline
int antiPlaneIntersect(Ray* r, Plane* p, Vec3* t, Vec3* normal, float* d, Vec2* uv){
	Vec3 diff = sub(p->point, r->origin);
	*d = dot(diff, p->normal) / dot(r->direction, p->normal);
	Vec3 ta = smul(r->direction, *d);
	*t = add(r->origin, ta);
	*normal = neg(p->normal);
	Vec3 planespace = sub(*t, p->point);
	*uv = {dot(planespace, p->right), dot(planespace, p->up)};
	return (*d > 0) && (dot(r->direction, *normal) < 0);
}

__device__ inline
int planeInside(Plane* plane, Vec3* p){
	return 0;
}

__device__ inline
int AABBIntersect(Ray* r, Geometry* AABB){
	//x
	float nx = AABB->min.x / r->direction.x;
	Vec3 nxp = smul(r->direction, nx);
	int nxtest = (nx > 0) && (nxp.y > AABB->min.y) && (nxp.y < AABB->max.y) && (nxp.z > AABB->min.z) && (nxp.z < AABB->max.z);
	
	float ax = AABB->max.x / r->direction.x;
	Vec3 axp = smul(r->direction, ax);
	int axtest = (ax > 0) && (axp.y > AABB->min.y) && (axp.y < AABB->max.y) && (axp.z > AABB->min.z) && (axp.z < AABB->max.z);
	
	//y
	float ny = AABB->min.y / r->direction.y;
	Vec3 nyp = smul(r->direction, ny);
	int nytest = (ny > 0) && (nyp.x > AABB->min.x) && (nyp.x < AABB->max.x) && (nyp.z > AABB->min.z) && (nyp.z < AABB->max.z);
	
	float ay = AABB->max.y / r->direction.y;
	Vec3 ayp = smul(r->direction, ay);
	int aytest = (ay > 0) && (ayp.x > AABB->min.x) && (ayp.x < AABB->max.x) && (ayp.z > AABB->min.z) && (ayp.z < AABB->max.z);
	
	//z
	float nz = AABB->min.z / r->direction.z;
	Vec3 nzp = smul(r->direction, nz);
	int nztest = (nz > 0) && (nzp.y > AABB->min.y) && (nzp.y < AABB->max.y) && (nzp.x > AABB->min.x) && (nzp.x < AABB->max.x);
	
	float az = AABB->max.z / r->direction.z;
	Vec3 azp = smul(r->direction, az);
	int aztest = (az > 0) && (azp.y > AABB->min.y) && (azp.y < AABB->max.y) && (azp.x > AABB->min.x) && (azp.x < AABB->max.x);
	
	return nxtest || axtest || nytest || aytest || nztest || aztest;
}

__device__ inline
int geoInside(Geometry* g, Vec3* p){
	switch(g->type){
		case GEO_PRIM:
			switch(g->prim->type){
				case PRIM_SPHERE:
					return sphereInside(&(g->prim->sphere), p);
				case PRIM_PLANE:
					return planeInside(&(g->prim->plane), p);
				default:
					return 0;
			}
			break;
		case GEO_SUB:
			return geoInside(g->subGeometry, p) && !geoInside(g->subGeometry + 1, p);
		default:
			return 0;
	}
}

__device__
int geoIntersect(Ray* r, Geometry* g, Vec3* t, Vec3* normal, float* d, Vec2* uv, Prim** prim, int depth);

__device__
void castRay(Ray* ray, float* d, Vec3* t, Vec3* normal, Prim** prim, Vec2* uv, int depth);

__device__ inline
int antiGeoIntersect(Ray* r, Geometry* g, Vec3* t, Vec3* normal, float* d, Vec2* uv, Prim** prim, int depth){
	int intersects;
	switch(g->type){
		case GEO_PRIM:
			*prim = g->prim;
			switch(g->prim->type){
				case PRIM_SPHERE:
					return antiSphereIntersect(r, &(g->prim->sphere), t, normal, d, uv);
				case PRIM_PLANE:
					return antiPlaneIntersect(r, &(g->prim->plane), t, normal, d, uv);
				default:
					return 0;
			}
			break;
		case GEO_SUB:
			intersects = antiGeoIntersect(r, g->subGeometry, t, normal, d, uv, prim, depth);
			if(!intersects) return 0;
			if(geoInside(g->subGeometry + 1, t)){
				intersects = geoIntersect(r, g->subGeometry + 1, t, normal, d, uv, prim, depth);
			}
			return intersects;
			break;
		default:
			return 0;
	}
}

__device__ inline
int geoIntersect(Ray* r, Geometry* g, Vec3* t, Vec3* normal, float* d, Vec2* uv, Prim** prim, int depth){
	int intersects, intersects2;
	switch(g->type){
		case GEO_PRIM:
			*prim = g->prim;
			switch(g->prim->type){
				case PRIM_SPHERE:
					return sphereIntersect(r, &(g->prim->sphere), t, normal, d, uv);
					break;
				case PRIM_PLANE:
					return planeIntersect(r, &(g->prim->plane), t, normal, d, uv);
					break;
				default:
					return 0;
			}
			break;
		case GEO_SUB:
			intersects = geoIntersect(r, g->subGeometry, t, normal, d, uv, prim, depth);
			if(!intersects) return 0;
			if(geoInside(g->subGeometry + 1, t)){
				intersects = antiGeoIntersect(r, g->subGeometry + 1, t, normal, d, uv, prim, depth);
				if(!geoInside(g->subGeometry, t)) return 0;
			}
			return intersects;
			break;
		case GEO_AABB:
			if(AABBIntersect(r, g)){
				*d = INFINITY;
				for(int i = 0; i < g->subGeometrySize; i++){
					Vec3 _t, _normal;
					float _d;
					Vec2 _uv;
					Prim* _prim;
					intersects2 = geoIntersect(r, g->subGeometry + i, &_t, &_normal, &_d, &_uv, &_prim, depth);
					if(intersects2 && (_d < *d)){
						*d = _d;
						*t = _t;
						*normal = _normal;
						*prim = _prim;
						*uv = _uv;
					}
				}
				return *d != INFINITY;
			}else{
				return 0;
			}
			break;
		case GEO_PORTAL:
			Vec3 _t, _normal;
			float _d;
			intersects2 = 0;
			Vec2 _uv;
			Ray newRay;
			switch(g->prim->type){
				case PRIM_SPHERE:
					intersects2 = antiSphereIntersect(r, &(g->prim->sphere), &_t, &_normal, &_d, &_uv);
					break;
				case PRIM_PLANE:
					intersects2 = planeIntersect(r, &(g->prim->plane), &_t, &_normal, &_d, &_uv);
					break;
				default:
					return 0;
			}
			if(!intersects2) return 0;
			if((depth > MAX_RAYS)){
				*d = _d;
				*t = _t;
				*normal = _normal;
				*prim = g->prim;
				*uv = _uv;
				return intersects2;
			}
			
			newRay.origin = add(_t, g->dest);
			newRay.direction = r->direction;
			castRay(&newRay, d, t, normal, prim, uv, depth + 1);
			if(*d == INFINITY){
				*d = _d;
				*prim = (Prim*)NULL;
				return 1;
			}else{
				*d = _d;
				return 1;
			}
			break;
		default:
			return 0;
	}
}

__device__
void castRay(Ray* ray, float* d, Vec3* t, Vec3* normal, Prim** prim, Vec2* uv, int depth){
	*d = INFINITY;
	
	for(int i = 0; i < worldSize; i++){
		Vec3 _t, _normal;
		float _d;
		int intersects;
		Vec2 _uv;
		Prim* _prim;
		intersects = geoIntersect(ray, world + i, &_t, &_normal, &_d, &_uv, &_prim, depth);
		if(intersects && (_d < *d)){
			*d = _d;
			*t = _t;
			*normal = _normal;
			*prim = _prim;
			*uv = _uv;
		}
	}
}

__device__ Color COLOR_SKY = {255, 240, 201, 255};
__device__ Color COLOR_WHITE = {255, 255, 255, 255};
__device__ Color COLOR_BLACK = {0, 0, 0, 255};

__device__
Color castColor(Ray* ray, int depth){
	Color outcolor;
	Vec3 antisun = neg(sunDir);
	
	Color colors[MAX_RAYS];
	float blendfactors[MAX_RAYS];
	int bounces = 0;
	Ray nextRay = *ray;
	
	colors[0] = {0, 0, 0, 0};
	
	for(int i = 0; i < MAX_RAYS; i++){
		float smallestd = INFINITY;
		Vec3 smallt, smallNormal;
		Prim* smallPrim;
		Vec2 uv;
		
		castRay(&nextRay, &smallestd, &smallt, &smallNormal, &smallPrim, &uv, 0);
		Color tex;
		Ray reflection;
		
		if(smallestd != INFINITY){
			//Hit sky through portal
			if(smallPrim == NULL){
				float sun = dot(ray->direction, antisun);
				sun = fmaxf(0, sun);
				Color skycolor = blend(COLOR_SKY, 1 - sun, COLOR_WHITE, sun);
				colors[i] = skycolor;
				break;
			}
			bounces++;
			if(smallPrim->type == PRIM_PLANE){
				int texselec = ((fabsf(fmodf(uv.x, 2.f)) - 1.f) * (fabsf(fmodf(uv.y, 2.f)) - 1.f)) < 0;
				if(uv.x < 0) texselec = !texselec;
				if(uv.y < 0) texselec = !texselec;
				if(texselec)
					tex = smallPrim->color;
				else
					tex = smallPrim->color2;
			}else{
				tex = smallPrim->color;
			}
			
			//Reflection
			Vec3 tiniNormal = smul(smallNormal, EPSILON);
			reflection.origin = add(smallt, tiniNormal);
			Vec3 dira = smul(smallNormal, 2 * dot(nextRay.direction, smallNormal));
			reflection.direction = sub(nextRay.direction, dira);
			
			blendfactors[i] = smallPrim->reflectivity;
			
			nextRay = reflection;
			colors[i] = tex;
			if(smallPrim->reflectivity == 0.f) break;
		}
		else{
			float sun = dot(ray->direction, antisun);
			sun = fmaxf(0, sun);
			Color skycolor = blend(COLOR_SKY, 1 - sun, COLOR_WHITE, sun);
			colors[i] = skycolor;
			break;
		}
	}
	
	
	outcolor = colors[bounces];
	
	
	for(int i = bounces - 1; i >= 0; i--){
		outcolor = blend(colors[i], 1.f - blendfactors[i], outcolor, blendfactors[i]);
	}
	
	//return (colors[0].a == 0) ? skycolor : colors[0];
	return outcolor;
}

__global__
void setPixel(int w, int h, Color* pixels, Vec3 pos, Vec2 look){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int x = index % w;
	int y = index / w;
	if(y <= h){
	
		float xn = (float)(x - (w/2)) / (float)(w/2);
		float yn = (float)(y - (h/2)) / (float)(w/2);
		
		xn *= xmult;
		yn *= ymult;
		
		Vec3 screenpos = {screendist, yn, xn};
		
		
		//screenpos = vecMatMul(lookMat, screenpos);
		screenpos = vecMatMul(matMul(rotatez(-look.y), rotatey(look.x)), screenpos);
		normalize(&screenpos);
		
		//screenpos = sub(screenpos, pos);
		
		Ray ray;
		ray.origin = pos;
		ray.direction = screenpos;
		pixels[index] = castColor(&ray, 0);
	}
}

int main(void)
{
	//Cuda init
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	cudaDeviceProp deviceProps;
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("\tDevice name: %s\n", prop.name);
		printf("\tMemory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		printf("\tMemory Bus Width (bits): %d\n", prop.memoryBusWidth);
		printf("\tPeak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
		printf("\tMax grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		printf("\tMax block size: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("\tMax threads per block: %d\n", prop.maxThreadsPerBlock);
		printf("\tCompute capability: %d.%d\n\n", prop.major, prop.minor);
		if(i == 0)
			deviceProps = prop;
	}
	cudaDeviceSetLimit(cudaLimitStackSize, 2048 * MAX_RAYS);
	size_t stackSize;
	cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
	printf("Max stack size: %zd\n", stackSize);
	
	//SDL init
	//The window we'll be rendering to
	SDL_Window* window = NULL;
	
	//The surface contained by the window
	SDL_Surface* screenSurface = NULL;

	//Initialize SDL
	if( SDL_Init( SDL_INIT_EVERYTHING ) < 0 )
	{
		printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
		return -1;
	}
	
	//Create window
	window = SDL_CreateWindow( "CUDA", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE);
	if( window == NULL )
	{
		printf( "Window could not be created! SDL_Error: %s\n", SDL_GetError() );
		return -1;
	}
	
	//Get window surface
	screenSurface = SDL_GetWindowSurface( window );
	
	printf("Surface format: %d\nDepth: %d\n", screenSurface->format->format, screenSurface->format->BitsPerPixel);\
	
	int blockSize = deviceProps.maxThreadsPerBlock / 2;
	int numBlocks = (screenSurface->w * screenSurface->h + blockSize - 1) / blockSize;
	
	#pragma region world
	
	Geometry* hostWorld;
	
	//int griddim = 4;
	
	
	//int hostWorldSize = (griddim * griddim * griddim) + 1;
	int hostWorldSize = 5;
	
	cudaMallocManaged(&hostWorld, hostWorldSize * sizeof(Geometry));
	
	//Create world
	{
		
		hostWorld[0].type = GEO_AABB;
		hostWorld[0].subGeometrySize = 2;
		hostWorld[0].min = {-1.5f, -1.f, -1.f};
		hostWorld[0].max = {1.5f, 1.f, 1.f};
		
		Geometry* subGeo;
		cudaMallocManaged(&subGeo, 2 * sizeof(Geometry));
		
		hostWorld[0].subGeometry = subGeo;
		
		subGeo[0].type = GEO_SUB;
		subGeo[0].subGeometrySize = 2;
		cudaMallocManaged(&(subGeo[0].subGeometry), sizeof(Geometry) * 2);
		
		subGeo[0].subGeometry[0].type = GEO_SUB;
		subGeo[0].subGeometry[0].subGeometrySize = 2;
		cudaMallocManaged(&(subGeo[0].subGeometry[0].subGeometry), sizeof(Geometry) * 2);
		subGeo[0].subGeometry[0].subGeometry[0].type = GEO_PRIM;
		cudaMallocManaged(&(subGeo[0].subGeometry[0].subGeometry[0].prim), sizeof(Prim));
		subGeo[0].subGeometry[0].subGeometry[0].prim->type = PRIM_SPHERE;
		subGeo[0].subGeometry[0].subGeometry[0].prim->color = {255, 50, 50, 255};
		subGeo[0].subGeometry[0].subGeometry[0].prim->reflectivity = 0.f;
		subGeo[0].subGeometry[0].subGeometry[0].prim->sphere = {{1.5f, 0.f, 0.f}, 1.f};
		
		subGeo[0].subGeometry[0].subGeometry[1].type = GEO_PRIM;
		cudaMallocManaged(&(subGeo[0].subGeometry[0].subGeometry[1].prim), sizeof(Prim));
		subGeo[0].subGeometry[0].subGeometry[1].prim->type = PRIM_SPHERE;
		subGeo[0].subGeometry[0].subGeometry[1].prim->color = {50, 50, 255, 255};
		subGeo[0].subGeometry[0].subGeometry[1].prim->reflectivity = 0.5f;
		subGeo[0].subGeometry[0].subGeometry[1].prim->sphere = {{1.5f, 0.f, 1.f}, 1.f};
		
		subGeo[0].subGeometry[1].type = GEO_PRIM;
		cudaMallocManaged(&(subGeo[0].subGeometry[1].prim), sizeof(Prim));
		subGeo[0].subGeometry[1].prim->type = PRIM_SPHERE;
		subGeo[0].subGeometry[1].prim->color = {50, 50, 255, 255};
		subGeo[0].subGeometry[1].prim->reflectivity = 0.5f;
		subGeo[0].subGeometry[1].prim->sphere = {{1.5f, 0.f, -1.f}, .5f};
		
		subGeo[1].type = GEO_PRIM;
		cudaMallocManaged(&(subGeo[1].prim), sizeof(Prim));
		subGeo[1].prim->type = PRIM_SPHERE;
		subGeo[1].prim->color = {255, 50, 255, 255};
		subGeo[1].prim->reflectivity = 0.5f;
		subGeo[1].prim->sphere = {{-1.5f, 0.f, 0.f}, 1.f};
		
	}
	
	hostWorld[1].type = GEO_PRIM;
	cudaMallocManaged(&(hostWorld[1].prim), sizeof(Prim));
	hostWorld[1].prim->type = PRIM_SPHERE;
	hostWorld[1].prim->color = {0, 0, 0, 255};
	hostWorld[1].prim->reflectivity = 1.f;
	hostWorld[1].prim->sphere = {{-3.f, -3.f, -3.f}, 1.5f};
	
	hostWorld[2].type = GEO_PRIM;
	cudaMallocManaged(&(hostWorld[2].prim), sizeof(Prim));
	hostWorld[2].prim->type = PRIM_PLANE;
	hostWorld[2].prim->color = {50, 50, 255, 255};
	hostWorld[2].prim->color2 = {222, 222, 222, 255};
	hostWorld[2].prim->reflectivity = 0.5f;
	hostWorld[2].prim->plane = {{-10.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}};
	
	hostWorld[3].type = GEO_PRIM;
	cudaMallocManaged(&(hostWorld[3].prim), sizeof(Prim));
	hostWorld[3].prim->type = PRIM_PLANE;
	hostWorld[3].prim->color = {255, 50, 50, 255};
	hostWorld[3].prim->color2 = {222, 222, 222, 255};
	hostWorld[3].prim->reflectivity = 0.5f;
	hostWorld[3].prim->plane = {{0.f, 1.f, 0.f}, {0.f, -1.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 0.f, 1.f}};
	
	hostWorld[4].type = GEO_PORTAL;
	cudaMallocManaged(&(hostWorld[4].prim), sizeof(Prim));
	hostWorld[4].prim->type = PRIM_SPHERE;
	hostWorld[4].prim->color = {0, 0, 0, 255};
	hostWorld[4].prim->reflectivity = 0.f;
	hostWorld[4].prim->sphere = {{3.f, 0.f, 3.f}, 1.f};
	hostWorld[4].dest = {0.f, -4.f, 0.f};
	
	
	cudaMemcpyToSymbol(world, &hostWorld, sizeof(Prim*));
	cudaMemcpyToSymbol(worldSize, &hostWorldSize, sizeof(int));
	
	#pragma endregion world
	
	float hostsd = 0.1;
	float hostxf = tan(FOV/2)*hostsd;
	float hostyf = tan(FOV/2)*hostsd;
	
	cudaMemcpyToSymbol(screendist, &hostsd, sizeof(float));
	cudaMemcpyToSymbol(xmult, &hostxf, sizeof(float));
	cudaMemcpyToSymbol(ymult, &hostyf, sizeof(float));
	
	for (int i = 0; i < SDL_NumJoysticks(); ++i) {
    if (SDL_IsGameController(i)) {
			printf("Joystick %i is supported by the game controller interface!\n", i);
		}
	}
	
	SDL_GameController* controller = SDL_GameControllerOpen(0);
	Vec3 pos = {10, -1, 0};
	Vec2 look = {FPI, 0};
	
	Color* pixels;// = (uint32_t*)screenSurface->pixels;
	cudaMallocManaged(&pixels, screenSurface->w * screenSurface->h * sizeof(Color));
	
	int quit = 0;
	int lastTime = SDL_GetTicks();
	float pastFPS[100];
	int fpsi = 0;
	SDL_Event e;
	printf("Before main loop\n");
	int oldSize = screenSurface->w * screenSurface->h;
	while(!quit){
		screenSurface = SDL_GetWindowSurface( window );
		while(SDL_PollEvent(&e) != 0){
			if(e.type == SDL_QUIT){
				quit = true;
			}else if(e.type == SDL_WINDOWEVENT_RESIZED){
				//This is fake
			}
		}
		
		//Detect window resize because events aren't fired properly
		int newSize = screenSurface->w * screenSurface->h;
		if(newSize != oldSize){
			numBlocks = (screenSurface->w * screenSurface->h + blockSize - 1) / blockSize;
			cudaFree(pixels);
			cudaMallocManaged(&pixels, screenSurface->w * screenSurface->h * sizeof(Color));
			oldSize = newSize;
		}
		
		setPixel<<<numBlocks, blockSize>>>(screenSurface->w, screenSurface->h, pixels, pos, look);
		gpuErrchk( cudaPeekAtLastError() );
		
		int nextTime = SDL_GetTicks();
		float dt = (float)(nextTime - lastTime) / 1000.f;
		float fps = 1/dt;
		lastTime = nextTime;
		
		pastFPS[fpsi++] = fps;
		if(fpsi >= 100) fpsi = 0;
		float avgfps = 0;
		for(int i = 0; i < 100; i++){
			avgfps += pastFPS[i];
		}
		avgfps /= 100.f;
		
		//This has a bunch of spaces to clear the rest of the line after it since it overwrites the same line over and over.
		printf("\rFPS: %.2f                                                    ", avgfps);
		
		//printf("Pos: %.2f, %.2f, %.2f\nLook: %.2f, %.2f\n", pos.x, pos.y, pos.z, look.x, look.y);
		int deadzone = 2000;
		int leftx = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTX);
		if(abs(leftx) < deadzone) leftx = 0;
		int lefty = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_LEFTY);
		if(abs(lefty) < deadzone) lefty = 0;
		int rightx = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTX);
		if(abs(rightx) < deadzone) rightx = 0;
		int righty = SDL_GameControllerGetAxis(controller, SDL_CONTROLLER_AXIS_RIGHTY);
		if(abs(righty) < deadzone) righty = 0;
		Vec3 move;
		move.x = ((float)-leftx) / 32767 * dt * sin(look.x) + ((float)-lefty) / 32767 * dt * cos(look.x);
		move.z = ((float)leftx) / 32767 * dt * cos(look.x) + ((float)-lefty) / 32767 * dt * sin(look.x);
		pos.x += move.x * MOVESPEED;
		pos.z += move.z * MOVESPEED;
		pos.y = 0;
		
		look.x += ((float)rightx) / 32767 * dt;
		look.y += ((float)righty) / 32767 * dt;
		
		gpuErrchk( cudaDeviceSynchronize() );
		SDL_memcpy(screenSurface->pixels, pixels, screenSurface->h * screenSurface->pitch);
		//Update the surface
		SDL_UpdateWindowSurface( window );
	}
	
	// Free memory
	cudaFree(pixels);
	cudaFree(hostWorld);
	
	//Destroy window
    SDL_DestroyWindow( window );

    //Quit SDL subsystems
    SDL_Quit();
	
	return 0;
}