#version 430 core

layout (location=0) in vec3 aPos;

layout(std430, binding=1) readonly buffer ssbo_data{
    float color_data[];
};

flat out vec3 aColor;
out float zPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 grid_size;
uniform int n_keys;

void main(){
    float cube_spacing = 0.1;
    int num_cubes = int(grid_size[2]*grid_size[1]*grid_size[0]);
    float red = 0.1;
    float green = color_data[gl_InstanceID];
    float blue = 0.1;
    
    if (n_keys>1) red = color_data[num_cubes+gl_InstanceID];
    if (n_keys>2) blue = color_data[2*num_cubes+gl_InstanceID];

    aColor = vec3(red,green,blue);

    int x = gl_InstanceID % int(grid_size[2]);
    int y = gl_InstanceID/int(grid_size[2]);
    y = y % int(grid_size[1]);
    int z = gl_InstanceID/(int(grid_size[2])*int(grid_size[1]));
    

    vec3 bPos = aPos;
    bPos.x = bPos.x+(1.0+cube_spacing)*float(x) - (grid_size[2]-1.0)/2.0;
    bPos.y = bPos.y+(1.0+cube_spacing)*float(y) - (grid_size[1]-1.0)/2.0;
    bPos.z = bPos.z+(1.0+cube_spacing)*float(z) - (grid_size[0]-1.0)/2.0;
    float zPos = bPos.z;

    gl_Position=projection*view*model*vec4(bPos,1.0);
}