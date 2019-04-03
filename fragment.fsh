#version 130
uniform sampler2D sTexture;
in vec2 vTextureCood;
out vec4 fragColor;

void main()
{
    fragColor = texture2D(sTexture,vTextureCood);
}
