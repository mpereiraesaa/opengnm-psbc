#version 450
/* The pixel half of the two-register distance contract: eight declared clip
 * distances, of which only the one in the SECOND packed register is read. The
 * register a read targets is the slot's own index (VARYING_SLOT_CLIP_DIST0/1),
 * not the attribute the linker assigned, so the description must name register
 * 1 and not register 0 - measuring this is what keeps the wrong-register
 * regression out of the fragment input list. */
in float gl_ClipDistance[8];
layout(location=0) out vec4 o_color;
void main() { o_color = vec4(gl_ClipDistance[4], 0.0, 0.0, 1.0); }
