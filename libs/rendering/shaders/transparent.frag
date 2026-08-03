#version 430 core

// shader outputs
layout (location = 0) out vec4 accum;
layout (location = 1) out float reveal;

flat in vec3 aColor; 
in float zPos;
in float Aalpha;


uniform float window_width;
uniform float window_height;
uniform float Ualpha;


float max(vec3 values) {
   float maximum = values[0];
   if (values[1]>maximum){
      maximum = values[1];
   }
   if (values[2]>maximum){
      maximum=values[2];
   }
   return maximum;
}


void main(){
   float alpha = max(aColor)>0.5?Ualpha:0.0;
   if (alpha < 0.1) discard;
   vec4 color = vec4(aColor, alpha);
   float weight = 
   //clamp(pow(min(1.0, color.a*10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - gl_FragCoord.zPos * 0.9, 3.0), 1e-2, 3e3);
   // max(min(1.0, max(max(color.r, color.g), color.b) * color.a), color.a) *
   //  clamp(0.03 / (1e-5 + pow(zPos / 200, 4.0)), 1e-2, 3e3);
   // color.a * clamp(10.0/(10e-5+pow(abs(zPos)/10.0,3)+pow(abs(zPos)/200.0,6)), 1e-2, 3e3);
   color.a * clamp(10.0/(10e-5+pow(abs(zPos)/10.0,2)+pow(abs(zPos)/400.0,6)), 1e-2, 3e3);
   accum = vec4(color.rgb*color.a, color.a) * weight;
   reveal = color.a;
}