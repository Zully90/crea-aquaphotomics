from fpdf import FPDF

def construct_report(fig, fig_name="figure"):
    import datetime
    import os
    import shutil
    
    PLOT_DIR = 'plots'
    
    isExist = os.path.exists(PLOT_DIR)

    if isExist == False:
        os.mkdir(PLOT_DIR)

    # Save visualization
    try:
        fig.write_image(f"{PLOT_DIR}/{fig_name}.png")
    except:
        fig.savefig(f"{PLOT_DIR}/{fig_name}.png")
        
    # Construct data shown in document
    pages_data = []
    temp = []
    # Get all plots
    files = os.listdir(PLOT_DIR)
    pages_data.append(temp)

    temp.append(f'{PLOT_DIR}/{fig_name}.png')

    return [*pages_data]

class PDF(FPDF):

    import os
    import shutil
    
    def __init__(self, target=None):
        self.target=target
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297


    def header(self):
        # Custom logo and positioning
        # Create an `assets` folder and put any wide and short image inside
        # Name the image `logo.png`
        self.image('assets/logo.png', 10, 8, 33)
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 1, f'Model report for {self.target.name}', 0, 0, 'R')
        self.ln(20)
        
    def footer(self):
        import datetime
        
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')
        self.cell(0, 10, f'Created at {str(datetime.datetime.now())[:16]}', 0, 0, 'R')

    def page_body(self, images):
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        if len(images) == 3:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
            self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
        elif len(images) == 2:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        else:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            
    def print_page(self, images):
        # Generates the report
        self.add_page()
        self.page_body(images)