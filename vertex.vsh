#version 330
uniform mat4 uPMatrix,uVMatrix,uMMatrix;
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCood;
smooth out vec2 vTextureCood;

void main(void)
{
    gl_Position = uPMatrix * uVMatrix * uMMatrix * vec4(aPosition,1);
    vTextureCood = aTexCood;
}
