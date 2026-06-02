#version 430 core

out vec4 FragColor;

flat in int InstanceID; 

uniform float window_width;
uniform float window_height;
uniform float[1000] color_data;

void main(){
   FragColor=vec4(color_data[InstanceID], 0.1, 0.1, 0.7); 
}