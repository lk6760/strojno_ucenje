import pickle
import requests, sys, os 
import random

random.seed(3)
	
pickle_path_all_genres = 'genreList_clipURLs.p'
pickle_path_filtered_genres = 'filteredAndCombinedGenreList.p'




def filter_and_combine_genres(pickle_file_path):
    with open(pickle_file_path, 'rb') as fp:
        #loading dictionary of genres and 30s clips
        genreList_clipURLs = pickle.load(fp)

    new_genreList_clipURLs = {}
    combinedGenreList = {}

    filtered_genres = [    #same as for GTZAN
        'blues',
        'classical',
        'country',
        'disco',
        'hip hop',
        'jazz',
        'metal',
        'pop',
        'reggae',
        'rock'
    ]

    ambigous_genres = [
        'italian pop rock',
        'blues rock',
        'indie pop rock',
        'german pop rock',
        'indonesian pop rock',
        'country rock',
        'czech pop rock',
        'danish pop rock',
        'reggae rock',
        'modern blues rock',
        'pop rock',
        'alternative pop rock',
        'spanish pop rock',
        'modern country rock',
        'jazz rock',
        'pop rock brasileiro',
        'jazz pop',
        'classic country pop',
        'country pop',
        'jazz metal',
        'neo classical metal',
        'classical jazz fusion',
        'jazz blues',
        'country blues'
    ]

    while len(filtered_genres) != 0:
        selected_genre = filtered_genres.pop()

        combinedGenreList[selected_genre] = list()
        
        #print("\n***** Filtering {} as genre *****\n".format(selected_genre))
        #selected_genre = selected_genre + " "
        for genres in genreList_clipURLs.keys():
            if selected_genre == "hip hop":
                if selected_genre in genres and genres not in ambigous_genres:
                    combinedGenreList[selected_genre].append(genres)

                    """
                    # check for conflicting genres with double or more genres
                    for second_genre in filtered_genres:
                        for word in genres.split():
                            if word == second_genre:
                                print(genres) 
                    """
            else:
                if genres not in ambigous_genres:
                    for word in genres.split():
                        if word == selected_genre:
                            combinedGenreList[selected_genre].append(genres)

                        """
                        # check for conflicting genres with double or more genres
                        for second_genre in filtered_genres:
                            for second_word in genres.split():
                                if second_word == second_genre:
                                    print(genres) 
                        """

    for cover_genre, genre_list in combinedGenreList.items():
        print("Selecting URLs for {}".format(cover_genre))
        new_genreList_clipURLs[cover_genre] = list()
        for old_genre in genre_list:
            for url in genreList_clipURLs[old_genre]:
                new_genreList_clipURLs[cover_genre].append(url)
        
        #print(len(list(new_genreList_clipURLs[cover_genre])))

        while len(list(new_genreList_clipURLs[cover_genre])) != 1640:
            random_song = random.choice(new_genreList_clipURLs[cover_genre])
            new_genreList_clipURLs[cover_genre].remove(random_song)

        #print(len(list(new_genreList_clipURLs[cover_genre])))

        #print("\n{}:\n{}".format(cover_genre, genre_list))


    with open('filteredAndCombinedGenreList.p', 'wb') as fp:
        pickle.dump(new_genreList_clipURLs, fp)




def download_mp3_files(pickle_file_path):
    with open(pickle_file_path, 'rb') as fp:
        #loading dictionary of genres and 30s clips
        genreList_clipURLs = pickle.load(fp)    

    for genreName, urlList in genreList_clipURLs.items():
        if genreName == 'hip hop':
            genreName = 'hiphop'
        i = 0
        parent_dir = os.getcwd()
        folder_path = os.path.join(parent_dir, genreName)
        if not(os.path.exists(folder_path)):
            print("Creating new folder " + genreName)
            os.mkdir(folder_path)
        
        for href in urlList:
            i += 1
            filename = genreName + "_clip_" + str(i) + ".mp3"
            save_path = os.path.join(folder_path, filename)
            if not(os.path.isfile(save_path)):
            	download = requests.get(href)
            	if download.status_code == 200:
            	    print(f"Downloading File {filename}")
            	    with open(save_path, 'wb') as f:
            	        f.write(download.content)
            	else:
            	    print(f"Download Failed For File {filename} ***********")
            else:
                print(save_path, " Already exists.")



download_mp3_files(pickle_path_filtered_genres)
