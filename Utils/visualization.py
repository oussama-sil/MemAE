import matplotlib.pyplot as plt




def plot_two_images(img1,img2,title_img1='Original',title_img2='Reconstruction',save=False,filename='./figure_original_reconstructed.png'):

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2)

    # Plot the first image in the first subplot
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title(title_img1)

    # Plot the second image in the second subplot
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title(title_img2)

    # Adjust spacing between subplots
    plt.tight_layout()


    if save:
        plt.savefig(filename)

    # Display the figure
    plt.show()



def plot_result(img,img_rec,img_rec_error,img_anom,title_img='Original',title_img_rec='Reconstruction',
                title_rec_error='Reconstruction error',title_img_anom='Detected anomalies',save=False,filename='./result.png'):
    # Create a figure with two rows and two columns of subplots
    fig, axes = plt.subplots(2, 2)

    # Flatten the axes array to iterate over it
    axes = axes.flatten()

    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(title_img)

    axes[1].imshow(img_rec, cmap='gray')
    axes[1].set_title(title_img_rec)

    axes[2].imshow(img_rec_error)
    axes[2].set_title(title_rec_error)

    axes[3].imshow(img_anom, cmap='gray')
    axes[3].set_title(title_img_anom)


    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure if save is True
    if save:
        plt.savefig(filename)

    # Display the figure
    plt.show()