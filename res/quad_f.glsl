#version 330

uniform sampler2D tex;

in vec2 f_uv;
out vec4 color;

void main(){
	color = texture(tex, f_uv);
}

