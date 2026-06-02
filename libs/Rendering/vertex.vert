#version 430 core

layout (location=0) in vec3 aPos;

flat out int InstanceID;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec3 grid_size;

void main(){
    InstanceID = gl_InstanceID;
    int x = gl_InstanceID % int(grid_size[0]);
    int y = gl_InstanceID/int(grid_size[0]);
    y = y % int(grid_size[1]);
    int z = gl_InstanceID/(int(grid_size[0])*int(grid_size[1]));
    int num_cubes = int(grid_size[0]*grid_size[1]*grid_size[2]);

    vec3 bPos = aPos;
    bPos.x = bPos.x+1.0*float(x);
    bPos.y = bPos.y+1.0*float(y);
    bPos.z = bPos.z+1.0*float(z);

    gl_Position=projection*view*model*vec4(bPos,1.0);
}