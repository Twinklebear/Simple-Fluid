#version 330

uniform mat4 mvp;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 v_uv;
out vec2 f_uv;

void main(){
	f_uv = v_uv.xy;
	gl_Position = mvp * vec4(pos, 1.f);
}

