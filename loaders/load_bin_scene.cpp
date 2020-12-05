#include "load_scene.h"
#include "load_scene_util.h"
#include "../bake_api.h"

#include <vector_types.h>
#include <optixu/optixu_matrix_namespace.h>

#include <cfloat>
#include <iostream>
#include <map>

namespace
{
	struct BinSceneMemory : public SceneMemory
	{
		BinSceneMemory() {}
		virtual ~BinSceneMemory() {}

		std::vector<bake::Mesh> meshes;
		std::vector<bake::Instance> instances;
	};

	struct BinSceneHeader
	{
		unsigned int num_meshes;
	};

	struct BinMeshHeader
	{
		float matrix[16];
		unsigned int num_vertices;
		unsigned int num_triangles;
	};

	void optimize_mesh(bake::Mesh& mesh)
	{
		typedef std::tuple<float, float, float, float, float, float> VertexKey;
		typedef std::map<VertexKey, unsigned int> VertexMap;

		VertexMap vertex_map;

		float3* vertices = reinterpret_cast<float3*>(mesh.vertices);
		float3* normals = reinterpret_cast<float3*>(mesh.normals);

		float3* new_vertices = new float3[mesh.num_vertices];
		float3* new_normals = new float3[mesh.num_vertices];

		const size_t num_indices = mesh.num_triangles * 3;
		unsigned int num_new_vertices = 0;

		for (size_t i = 0; i < num_indices; ++i) {
			unsigned int vertex_index = mesh.tri_vertex_indices[i];
			const float3& vertex = vertices[vertex_index];
			const float3& normal = normals[vertex_index];

			VertexKey vertex_key = { vertex.x, vertex.y, vertex.z, normal.x, normal.y, normal.z };
			VertexMap::const_iterator pos = vertex_map.find(vertex_key);

			if (pos == vertex_map.end()) {
				vertex_index = num_new_vertices;

				new_vertices[vertex_index] = vertex;
				new_normals[vertex_index] = normal;

				vertex_map[vertex_key] = vertex_index;
				mesh.tri_vertex_indices[i] = vertex_index;

				++num_new_vertices;
			}
			else {
				mesh.tri_vertex_indices[i] = pos->second;
			}
		}

		delete[] mesh.vertices;
		delete[] mesh.normals;

		mesh.vertices = reinterpret_cast<float*>(&new_vertices[0]);
		mesh.normals = reinterpret_cast<float*>(&new_normals[0]);
		mesh.num_vertices = num_new_vertices;
	}

	void load_bin_mesh(FILE* file, bake::Mesh& mesh, bake::Instance& instance, unsigned int mesh_index, float scene_bbox_min[3], float scene_bbox_max[3])
	{
		BinMeshHeader header;
		fread(&header, sizeof(header), 1, file);

		float* vertices = new float[header.num_vertices * 3];
		fread(vertices, sizeof(float) * header.num_vertices * 3, 1, file);

		float* normals = new float[header.num_vertices * 3];
		fread(normals, sizeof(float) * header.num_vertices * 3, 1, file);

		unsigned int* indices = new unsigned int[header.num_triangles * 3];
		fread(indices, sizeof(unsigned int) * header.num_triangles * 3, 1, file);

		// Setup mesh

		mesh.num_vertices = header.num_vertices;
		mesh.vertices = &vertices[0];
		mesh.vertex_stride_bytes = 0;
		mesh.normals = &normals[0];
		mesh.normal_stride_bytes = 0;
		mesh.num_triangles = header.num_triangles;
		mesh.tri_vertex_indices = &indices[0];

		optimize_mesh(mesh);	// TODO: Add a flag in the binary file for toggling optimization of mesh

		std::fill(mesh.bbox_min, mesh.bbox_min + 3, FLT_MAX);
		std::fill(mesh.bbox_max, mesh.bbox_max + 3, -FLT_MAX);
		for (size_t i = 0; i < mesh.num_vertices; ++i) {
			expand_bbox(mesh.bbox_min, mesh.bbox_max, &mesh.vertices[3 * i]);
		}

		// Setup instance

		instance.mesh_index = mesh_index;
		instance.storage_identifier = mesh_index;

		const optix::Matrix4x4 xform = optix::Matrix4x4(header.matrix);
		std::copy(xform.getData(), xform.getData() + 16, instance.xform);

		// Expand scene bbox to encompass the mesh bbox

		xform_bbox(xform, mesh.bbox_min, mesh.bbox_max, instance.bbox_min, instance.bbox_max);
		expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_min);
		expand_bbox(scene_bbox_min, scene_bbox_max, instance.bbox_max);
	}
}


bool load_bin_scene(const char* filename, bake::Scene& scene, float scene_bbox_min[3], float scene_bbox_max[3], SceneMemory*& base_memory, size_t num_instances_per_mesh)
{
	FILE* file = fopen(filename, "rb");
	if (!file) return false;

	BinSceneMemory* memory = new BinSceneMemory();

	std::cerr << "Opening bin-file..." << "\n";

	BinSceneHeader header;
	fread(&header, sizeof(header), 1, file);

	memory->meshes.resize(header.num_meshes);
	memory->instances.resize(header.num_meshes);

	for (unsigned int n = 0; n < header.num_meshes; ++n)
	{
		load_bin_mesh(file, memory->meshes[n], memory->instances[n], n, scene_bbox_min, scene_bbox_max);
	}

	scene.meshes = &memory->meshes[0];
	scene.num_meshes = memory->meshes.size();
	scene.instances = &memory->instances[0];
	scene.num_instances = memory->instances.size();
	base_memory = memory;

	return true;
}
