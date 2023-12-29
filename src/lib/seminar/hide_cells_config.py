c = get_config()
c.TagRemovePreprocessor.enabled = True
c.TagRemovePreprocessor.remove_cell_tags = ('hide_cell',)
c.TemplateExporter.exclude_input_prompt = True
c.TemplateExporter.exclude_output_prompt = True
