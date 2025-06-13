# 📸 Screenshot Guide for LLM Factory

This guide helps you capture the right screenshots for the README documentation.

## Required Screenshots

### 1. `dashboard-main.png`
- **What to capture**: Full browser window showing the main LLM Factory dashboard
- **Size**: 1920x1080 or similar wide format
- **Focus**: Show the header with "LLM Factory" title and navigation tabs
- **Notes**: Capture with the first tab (Raw Data) active

### 2. `pipeline-execution.png`
- **What to capture**: Terminal/command prompt showing the execution of `run_simple_finetuning.py`
- **Size**: Terminal window showing JSON output
- **Focus**: Show the detailed statistics and progress steps
- **Notes**: Include the structured JSON output at the end

### 3. `interface-overview.png`
- **What to capture**: Full interface showing multiple tabs and content
- **Size**: Full browser window
- **Focus**: Show navigation pills and content area with charts
- **Notes**: Switch between tabs to show different sections

### 4. `raw-data-analysis.png`
- **What to capture**: Raw Data tab content
- **Size**: Content area with charts and statistics
- **Focus**: Show data table, statistics, and histogram charts
- **Notes**: Scroll to show text length analysis charts

### 5. `processed-dataset.png`
- **What to capture**: Processed Dataset tab content
- **Size**: Content area showing split statistics and charts
- **Focus**: Show dataset splits and per-split analysis
- **Notes**: Show the tabbed interface for train/validation/test

### 6. `tokenization.png`
- **What to capture**: Tokenization tab content
- **Size**: Content area showing token analysis
- **Focus**: Show tokenization example and statistics
- **Notes**: Include token length distribution charts

### 7. `model-architecture.png`
- **What to capture**: Model Architecture tab content
- **Size**: Content area showing model details
- **Focus**: Show model configuration and parameter details
- **Notes**: Include the layer breakdown table

### 8. `training-process.png`
- **What to capture**: Training Process tab content
- **Size**: Content area showing training metrics
- **Focus**: Show loss curves and training statistics
- **Notes**: Include both training and evaluation loss charts

### 9. `inference-results.png`
- **What to capture**: Inference Results tab content
- **Size**: Content area showing prediction results
- **Focus**: Show sample predictions and confidence analysis
- **Notes**: Include the confidence distribution chart

## Screenshot Requirements

### Technical Specifications
- **Format**: PNG (preferred) or JPG
- **Resolution**: Minimum 1200px width
- **Quality**: High quality, clear text
- **Browser**: Use Chrome or Firefox for consistent rendering

### Content Guidelines
- **Clean Interface**: Close unnecessary browser tabs/windows
- **Good Lighting**: Ensure screenshots are bright and clear
- **Text Readability**: All text should be clearly readable
- **Complete Views**: Don't crop important content

### File Naming
- Use exact names as listed above
- All lowercase with hyphens
- .png extension
- Place in `docs/screenshots/` directory

## Capture Process

1. **Start the Application**
   ```bash
   python app.py
   ```

2. **Open Browser**
   - Navigate to `http://127.0.0.1:5000`
   - Set browser zoom to 100%
   - Use full-screen mode (F11) for clean captures

3. **Capture Each Tab**
   - Click through each navigation tab
   - Wait for content to load completely
   - Take screenshot of each section

4. **Terminal Screenshots**
   - Run `python run_simple_finetuning.py`
   - Capture the detailed output
   - Show JSON statistics at the end

## Tools for Screenshots

### Windows
- **Built-in**: Windows + Shift + S
- **Snipping Tool**: More advanced cropping
- **PowerToys**: For consistent sizing

### macOS
- **Built-in**: Cmd + Shift + 4
- **Preview**: For editing and annotation
- **CleanMyMac X**: For enhanced captures

### Linux
- **GNOME Screenshot**: Built-in tool
- **Flameshot**: Feature-rich screenshot tool
- **Shutter**: Advanced screenshot editor

## Post-Processing

### Recommended Edits
- **Crop**: Remove unnecessary browser chrome
- **Resize**: Ensure consistent dimensions
- **Compress**: Optimize file size without quality loss
- **Annotate**: Add arrows or highlights if needed

### Tools
- **Online**: TinyPNG for compression
- **Desktop**: GIMP, Photoshop, or Preview
- **Automated**: ImageOptim (macOS) or similar

## Verification Checklist

- [ ] All 9 screenshots captured
- [ ] Files named correctly
- [ ] Placed in correct directory
- [ ] Text is readable at normal viewing size
- [ ] No sensitive information visible
- [ ] Consistent browser/theme appearance
- [ ] All charts and visualizations visible
- [ ] Screenshots show actual data, not placeholder text

## Notes

- Screenshots should represent the actual working application
- Ensure all features are demonstrated
- Use consistent browser theme/appearance
- Update screenshots when UI changes significantly
- Consider creating GIFs for interactive features

---

After capturing screenshots, update the README.md file paths if needed and commit all files to the repository.
