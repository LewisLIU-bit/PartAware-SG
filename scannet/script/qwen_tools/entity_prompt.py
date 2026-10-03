PROMPT_VERSION = "qwen_entity_categories_v1"

ENTITY_PROMPT = """
Analyze the supplied image and identify visible physical object categories
for an open-vocabulary 3D scene mapping pipeline.

Discover categories from the image. Do not restrict your answer to a
predefined category list.

Selection rules:
1. Include clearly visible physical objects, including partially visible
   objects when their category is reasonably identifiable.
2. Include vegetation such as trees and bushes, and physical structures
   such as buildings, walls, fences, and signs when visible.
3. Do not report sky, shadows, reflections, glare, printed text, logos,
   drawings, or surface patterns as separate physical objects.
4. Report a billboard or sign as its physical carrier. Do not report objects
   depicted in its printed image as objects present in the scene.
5. Do not report individual zebra-crossing stripes, lane markings, or
   repeated surface textures as separate objects.
6. Prefer a whole object over its parts. For example, do not separately
   report the wheels of a reported car or the windows of a reported building.
7. For this version, report roads, sidewalks, and other continuous ground
   surfaces separately in "surface_categories". Do not mix them with objects.
8. Do not infer objects that are hidden or outside the image.
9. Use short, common English category names suitable for text-guided
   detection. Avoid brand names, overly specific guesses, and synonyms
   that duplicate the same category.
10. Return each category only once, even if multiple instances are visible.

Description rules:
- Describe only visible appearance.
- Descriptions apply to a category within this image, not to a uniquely
  identified instance.
- If multiple instances differ in appearance, briefly describe that
  variation instead of assigning one instance's properties to all of them.
- Do not invent dimensions, materials, distances, or object identities.

Return only a valid JSON object, without Markdown fences or additional text:
{
  "objects": [
    {
      "name": "short English category name",
      "description": "Brief description grounded in the supplied image."
    }
  ],
  "surface_categories": [
    {
      "name": "short English surface category name",
      "description": "Brief description grounded in the supplied image."
    }
  ]
}

Use empty arrays when no suitable categories are visible.
"""


INDOOR_PROMPT_VERSION = "qwen_indoor_entities_v2"

INDOOR_ENTITY_PROMPT = """
Identify clearly visible physical object categories in this indoor image.
These categories will be passed to an object detector and then an instance
segmenter to construct a 3D object map.

Object selection:
1. Include recognizable furniture, appliances, equipment, and small objects.
   Include partially visible objects only when their category is supported
   by visible evidence.
2. Put walls, floors, ceilings, and continuous architectural surfaces in
   "surface_categories", never in "objects".
3. Do not label an uncertain large foreground region as an object merely
   because its shape resembles a table, cabinet, or appliance.
   Omit uncertain categories rather than guessing.
4. Do not report reflections, illumination, shadows, printed pictures,
   logos, text, or surface patterns as physical objects.
5. Prefer a complete object over its components. Do not separately report
   speaker cones, buttons, panels, or legs of an already reported object.
6. Distinguish speakers from audio amplifiers using visible evidence.
   Do not infer an amplifier merely because speakers are present.
7. Avoid overlapping category names for the same object. Use one suitable
   category rather than listing both a broad category and its subtype
   for the same visible instance.
8. A rug is an object only when a distinct rug is visibly identifiable.
   Do not rename an uncertain floor region as a rug.
9. Use short, common English category names. Return each category once,
   even if several instances are visible.
10. Do not infer hidden or off-screen objects.

Descriptions:
- Describe only visible appearance.
- A description applies to the category in this frame, not a unique instance.
- If instances differ, briefly describe their visible variation.
- Do not invent materials, dimensions, distances, or object identities.

Return only valid JSON:
{
  "objects": [
    {
      "name": "short English category name",
      "description": "Brief description supported by visible evidence."
    }
  ],
  "surface_categories": [
    {
      "name": "short English surface category name",
      "description": "Brief description supported by visible evidence."
    }
  ]
}

Use empty arrays when appropriate.
"""


CANDIDATE_REVIEW_VERSION = "candidate_review_v1"

CANDIDATE_REVIEW_PROMPT = """
Inspect the target region in this composite image.
LEFT: the original scene with a red target rectangle.
RIGHT: an enlarged crop of that region with some surrounding context.
Both panels show the SAME observation, not two different objects.

Describe what is actually visible in the target rectangle. No detector
category is provided. Do not infer the answer from an expected category.

Rules:
1. Identify the main physical object targeted by the rectangle.
2. A device resting on furniture is an independent object, not a component
   of the furniture. Distinguish this from an actual attached component.
3. If the rectangle targets only a fragment of an object, record "fragment".
   If an object is occluded or cut off, record partial visibility.
   Partial visibility alone does not make it an attached component.
4. If several independent objects are targeted with no clear main object,
   use scope="multiple_objects". Do not describe them as one instance.
5. Appearance must describe only the target's visible shape, color,
   surfaces and distinctive details. Exclude location, nearby objects,
   and uncertain material or functional claims from appearance.
6. Put spatial information only in relations. The subject is the target.
   Use an empty list when the relation or anchor cannot be established.
   A 2D overlap is not evidence of physical containment.
7. Use a broad category when function is uncertain. Do not invent a brand,
   model, hidden structure, or invisible object.
8. Background, reflections and uncertain regions must be marked explicitly.
9. For front/behind, use the camera viewpoint and record reference="camera".
   For other relations use reference="scene".

Return only a JSON object:
{
  "scope": "single_object|multiple_objects|background|uncertain",
  "category": "short English category, or an empty string if unknown",
  "appearance": "visible appearance of the target only",
  "visibility": "full|partial|uncertain",
  "box_fit": "tight|loose|fragment|uncertain",
  "object_role": "independent|component|uncertain",
  "relations": [
    {
      "predicate": "on|inside|below|above|in_front_of|behind",
      "anchor": "short description of the reference object",
      "reference": "scene|camera",
      "evidence": "brief visible evidence for this relation"
    }
  ]
}
"""

INSTANCE_LOCATION_VERSION = "qwen_instance_location_v1"

INSTANCE_LOCATION_PROMPT = """
Locate individual visible physical objects in this indoor image.

Rules:
1. Return ONE record per physical object. Multiple objects may have the
   same category name. Never combine several objects into one record.
2. Examine the whole image, then inspect visible objects on shelves,
   beneath furniture tops, and within open furniture compartments.
   Include only objects supported by visible evidence.
3. Equipment resting on furniture is an independent object.
   Do not treat it as a component of the furniture.
4. Do not separately report buttons, handles, drivers, legs, or panels
   of an object already reported.
5. Exclude floors, walls, ceilings, shadows, reflections, printed images,
   and illumination patterns.
6. Do not assume an expected number or category of objects.
7. Use a broad category if the exact function is uncertain.
8. For a partially visible object, bound its visible extent and mark
   visibility="partial". Do not invent hidden extents.
9. Appearance must describe ONLY this object's visible features.
   Use at most 20 English words. Exclude other objects and spatial relations.

Bounding boxes:
- Use [xmin, ymin, xmax, ymax].
- Coordinates are normalized to 0..1000.
- Top-left of the image is (0, 0); bottom-right is (1000, 1000).
- Give a tight box around each individual object's visible extent.
- Do not use a furniture-sized box for an object resting on that furniture.

Return only valid JSON:
{
  "coordinate_system": "xyxy_1000",
  "objects": [
    {
      "name": "short English category name",
      "appearance": "short description of this individual object",
      "visibility": "full",
      "bbox_xyxy": [100, 200, 300, 400]
    }
  ]
}

Allowed visibility values: full, partial, uncertain.
The example coordinates are illustrative; determine coordinates from the image.
Return an empty objects list if no suitable objects are visible.
"""