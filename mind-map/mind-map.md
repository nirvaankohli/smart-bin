
- Smart Bin

  - Hardware

    - Raspberry Pi

      - Need to set up
        - Step-by-step guides for connecting sensors to Raspberry Pi
        - Online tutorials for Raspberry Pi setup
        - Programming Raspberry Pi for Smart Bin functionalities
        - Troubleshooting common Raspberry Pi setup issues
        - Might need to buy setup materials
        - Hook it up to both motors & camera
      - Energy-efficient programming
        - Power-saving features in software design
        - Utilize sleep mode to conserve power when not in use
        - Consider low-power consumption peripherals
      - Find a way to implement power supply seamlessly in design (cable required)
        - Investigate using hidden cable channels for clean look
        - Utilize magnetic connectors for cable management
      - Experiment with other related smart-bin use cases
        - Use it remotely 
        - Have the bin do other fun stuff
  
    - Motors

      - Make the trash bin open
        - Motors
          - Links
            - Option 1: 
            - Option 2: 
          - Requirements
            - Motor needs to be compatible with Raspberry Pi
            - Motor needs to be powerful enough
            - Motor needs to be in a small form factor
            - Should be priced reasonably
        - Foot pedal mechanism for hands-free operation (human-powered backup)
    - Trash Bin

      - MVP
        - Body for Bin
          - Divider in the middle of the bin
          - Will be Cardboard 
          - Might change later on
        - Flaps(Prob Same as In Testing one)
      - In testing: Two cardboard flaps as bins
        - Easy to replace/modify for iterations
        - Cost-effective DIY solution for initial testing
        - Compartment for storing extra trash bags
        - Design flaps for easy access and durability
        - Sustainability aspect considered
    - Web Cam / Camera  
      - What Makes a good camera?
        - Find a compatible camera for Raspberry Pi
        - Resolution and image quality
        - Low-light performance and night vision capabilities
        - Price point and affordability
        - Ease of setup and integration with Raspberry Pi
        - Compatibility with Raspberry Pi models
      - Links
        - research later
        - research later 2

  - Software 
    - Models
        - Model Training
          - Collect and preprocess data for training(Find a Dataset)
          - Research past data archetypes that did well on the dataset
          - Train and validate models for smart bin functionalities
          - Find the sweet spot between amount of compute used & accuracy
            - The
        - Documentation and Reasoning
          - Create detailed reasoning papers explaining model choices
          - Document training process and results
          - Maintain clear records for reproducibility
        - Optimizing runtime
        - Trash Detection(Crop the image to feed the model)
    - Website Demo
      - Develop a web-based interface to showcase smart bin features
      - Provide live data visualization and user interaction
      - Ensure responsive design for various devices
    - API / Hosting on Raspberry Pi
      - Raspberry Pi
        - Build RESTful APIs to interact with smart bin hardware and software

      - HTTPS RESTful API
        - Ensure secure and efficient communication between components
